"""
server.py — fundus-mcp-vascx
=============================
FastMCP server exposing VascX artery/vein segmentation and fovea
localization as MCP tools, deployed via Prefect Horizon.

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Expected weight files:
    weights/av_july24.pt        AV segmentation ensemble
    weights/fovea-july24.pt     Fovea localization heatmap regression

Tools:
    segment_av(image_b64, image_id)     → AV mask stats + base64 NPZ  [background task]
    localize_fovea(image_b64, image_id) → fovea (x, y) coordinates    [background task]
    health()                            → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(__file__).parent / "weights"
AV_WEIGHTS    = WEIGHTS_DIR / "av_july24.pt"
FOVEA_WEIGHTS = WEIGHTS_DIR / "fovea-july24.pt"
TMP_CACHE_DIR = Path("/tmp/vascx-model-cache")

# Redis is required for background tasks to work across stateless HTTP requests.
# Each request may hit a different process; in-memory task state won't survive.
# Set FASTMCP_DOCKET_URL=rediss://... in your Horizon environment variables.
_redis_url = os.environ.get("FASTMCP_DOCKET_URL")
if not _redis_url:
    logger.warning(
        "FASTMCP_DOCKET_URL is not set — background tasks will fail across "
        "stateless HTTP requests. Set this to your Redis URL in Horizon env vars."
    )

# ---------------------------------------------------------------------------
# Model loader with /tmp caching
# ---------------------------------------------------------------------------

_runner = None


def _get_runner():
    global _runner
    if _runner is not None:
        return _runner

    import cv2
    import torch
    cv2.setNumThreads(1)

    if not AV_WEIGHTS.exists():
        raise FileNotFoundError(f"AV weights not found: {AV_WEIGHTS}")
    if not FOVEA_WEIGHTS.exists():
        raise FileNotFoundError(f"Fovea weights not found: {FOVEA_WEIGHTS}")

    from rtnls_inference import SegmentationEnsemble, HeatmapRegressionEnsemble
    from rtnls_fundusprep.preprocessor import parallel_preprocess

    # Use /tmp cache if available — avoids re-parsing .pt files on cold starts
    av_src    = str(TMP_CACHE_DIR / "av_july24.pt")    if (TMP_CACHE_DIR / "av_july24.pt").exists()    else str(AV_WEIGHTS)
    fovea_src = str(TMP_CACHE_DIR / "fovea-july24.pt") if (TMP_CACHE_DIR / "fovea-july24.pt").exists() else str(FOVEA_WEIGHTS)

    if av_src != str(AV_WEIGHTS):
        logger.info(f"Loading AV model from /tmp cache (PID={os.getpid()}) ...")
    else:
        logger.info(f"Loading AV model from weights/ (PID={os.getpid()}) ...")

    class _Runner:
        seg        = SegmentationEnsemble.from_file(av_src)
        hm         = HeatmapRegressionEnsemble.from_file(fovea_src)
        preprocess = staticmethod(parallel_preprocess)

    _runner = _Runner()

    # Save to /tmp so subsequent cold starts in this container load faster
    if not TMP_CACHE_DIR.exists():
        logger.info(f"Saving parsed weights to {TMP_CACHE_DIR} ...")
        TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(_runner.seg,  str(TMP_CACHE_DIR / "av_july24.pt"))
        torch.save(_runner.hm,   str(TMP_CACHE_DIR / "fovea-july24.pt"))
        logger.info("Cache saved.")

    logger.info("VascX models ready.")
    return _runner


# Pre-warm only when running as a real server, not during `fastmcp inspect`.
# inspect imports the module to discover tools but doesn't start uvicorn,
# so we skip the heavy opencv/torch imports to avoid libxcb/X11 errors in
# the Horizon build container.
_is_inspect = (
    os.environ.get("FASTMCP_INSPECT") == "1"
    or "inspect" in " ".join(sys.argv)
)
if not _is_inspect:
    logger.info(f"Pre-warming VascX models at module import (PID={os.getpid()}) ...")
    _get_runner()
    logger.info("Models ready.")
else:
    logger.info("Skipping model pre-warm during fastmcp inspect (build time).")

# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("fundus-vascx")

# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------

def _preprocess(image_b64: str, image_id: str, tmp: Path):
    """Decode image, run parallel_preprocess, return (rgb_img, ce_img, bounds)."""
    import cv2
    from PIL import Image as _Image

    cv2.setNumThreads(1)
    runner = _get_runner()

    rgb_dir  = tmp / "rgb"; rgb_dir.mkdir()
    ce_dir   = tmp / "ce";  ce_dir.mkdir()
    img_path = tmp / f"{image_id}.jpg"
    img_path.write_bytes(base64.b64decode(image_b64))

    bounds_list = runner.preprocess(
        [img_path], rgb_path=rgb_dir, ce_path=ce_dir, n_jobs=1,
    )
    result = bounds_list[0]
    if not result.get("success", True):
        raise RuntimeError(f"Preprocessing failed: {result}")

    rgb_img = _Image.open(rgb_dir / f"{image_id}.png").convert("RGB")
    ce_img  = _Image.open(ce_dir  / f"{image_id}.png").convert("RGB")
    return rgb_img, ce_img, result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(task=TaskConfig(mode="required", poll_interval=timedelta(seconds=5)))
async def segment_av(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
    """
    Run VascX artery/vein segmentation on a fundus image.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Stem used for output filenames (e.g. "retina").

    Returns:
        JSON with artery/vein pixel counts, crop bounds, and base64-encoded
        NPZ containing artery, vein, and av_raw mask arrays.
    """
    import numpy as np
    from datetime import datetime

    await progress.set_total(3)

    try:
        await progress.set_message("Loading models...")
        runner = _get_runner()

        await progress.set_message("Preprocessing image...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                rgb_img, ce_img, bounds = _preprocess(image_b64, image_id, tmp)
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})
            await progress.increment()

            await progress.set_message("Running AV segmentation...")
            av_pred = runner.seg.predict_preprocessed(
                [np.array(rgb_img)], [np.array(ce_img)]
            )[0]
            await progress.increment()

            await progress.set_message("Computing mask statistics...")
            av_raw  = av_pred.astype(np.uint8)
            rgb_arr = np.array(rgb_img)
            av_raw[rgb_arr.max(axis=2) <= 10] = 0

            _A, _V, _X = 1, 2, 3
            artery = ((av_raw == _A) | (av_raw == _X)).astype(np.uint8)
            vein   = ((av_raw == _V) | (av_raw == _X)).astype(np.uint8)

            b      = bounds.get("bounds", {})
            cx     = b.get("center", [0, 0])[0]
            cy     = b.get("center", [0, 0])[1]
            r      = b.get("radius", 0)
            orig_h, orig_w = b.get("hw", (0, 0))

            buf = io.BytesIO()
            np.savez_compressed(buf, artery=artery, vein=vein, av_raw=av_raw)

            payload = json.dumps({
                "success":              True,
                "image_id":             image_id,
                "shape":                list(av_raw.shape),
                "artery_pixel_count":   int(artery.sum()),
                "vein_pixel_count":     int(vein.sum()),
                "crossing_pixel_count": int((av_raw == _X).sum()),
                "vessel_pixel_count":   int((av_raw >= 1).sum()),
                "crop_bounds": {
                    "cx": cx, "cy": cy, "radius": r,
                    "orig_h": orig_h, "orig_w": orig_w,
                    "x1": max(0, cx - r), "y1": max(0, cy - r),
                    "x2": min(orig_w, cx + r), "y2": min(orig_h, cy + r),
                },
                "masks_b64":  base64.b64encode(buf.getvalue()).decode(),
                "model":      AV_WEIGHTS.name,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            logger.info(f"AV payload size: {len(payload)/1024:.1f} KB")
            await progress.increment()
            return payload

    except Exception as e:
        logger.error(f"segment_av failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool(task=TaskConfig(mode="required", poll_interval=timedelta(seconds=5)))
async def localize_fovea(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
    """
    Localize the fovea (macula center) in a fundus image using VascX.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with fovea x, y coordinates in preprocessed image space.
    """
    import numpy as np

    await progress.set_total(3)

    try:
        await progress.set_message("Loading models...")
        runner = _get_runner()

        await progress.set_message("Preprocessing image...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                rgb_img, ce_img, _ = _preprocess(image_b64, image_id, tmp)
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})
            await progress.increment()

            await progress.set_message("Localizing fovea...")
            hm_pred = runner.hm.predict_preprocessed(
                [np.array(rgb_img)], [np.array(ce_img)]
            )[0]
            await progress.increment()

            y, x = np.unravel_index(np.argmax(hm_pred), hm_pred.shape)

            await progress.set_message("Done.")
            await progress.increment()

            return json.dumps({
                "success":  True,
                "image_id": image_id,
                "x":        float(x),
                "y":        float(y),
                "model":    FOVEA_WEIGHTS.name,
            })

    except Exception as e:
        logger.error(f"localize_fovea failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports weight file status and /tmp cache presence."""
    return json.dumps({
        "status":  "ok",
        "service": "fundus-vascx",
        "weights": {
            "artery_vein":  str(AV_WEIGHTS),
            "fovea":        str(FOVEA_WEIGHTS),
            "av_exists":    AV_WEIGHTS.exists(),
            "fovea_exists": FOVEA_WEIGHTS.exists(),
        },
        "tmp_cache": TMP_CACHE_DIR.exists(),
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
