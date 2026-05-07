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

import subprocess
import sys

# ---------------------------------------------------------------------------
# Force headless OpenCV before any other import touches cv2.
# retinalysis-fundusprep/vascx pulls in opencv-python (GUI variant) as a
# transitive dependency, which requires libxcb.so.1 (X11) — not available
# in the Horizon runtime container. We force-reinstall the headless variant
# here, before cv2 is imported anywhere, so the correct shared library is used.
# ---------------------------------------------------------------------------
def _ensure_headless_opencv():
    try:
        import importlib.util
        spec = importlib.util.find_spec("cv2")
        if spec is None:
            return  # not installed yet, requirements.txt will handle it
        cv2_path = spec.origin or ""
        if "headless" not in cv2_path:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "--force-reinstall", "--no-deps", "opencv-python-headless",
            ])
            # Invalidate import caches so next cv2 import gets the headless build
            import importlib
            importlib.invalidate_caches()
    except Exception as e:
        print(f"[WARNING]: opencv headless check failed: {e}", file=sys.stderr)

_ensure_headless_opencv()

# ---------------------------------------------------------------------------
# Ensure retinalysis-inference is importable.
# Its tight dependency pins (huggingface-hub==0.25.1, albumentations==1.3.1
# etc.) may cause pip to skip or partially install it when resolving the full
# requirements.txt. Install it with --no-deps as a fallback so the inference
# classes are always available regardless of pip resolution order.
# ---------------------------------------------------------------------------
def _ensure_rtnls_inference():
    try:
        import rtnls_inference  # noqa: F401 — already installed, nothing to do
    except ImportError:
        print("[INFO]: rtnls_inference not found, installing retinalysis-inference --no-deps ...", file=sys.stderr)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "--no-deps", "retinalysis-inference",
            ])
            import importlib
            importlib.invalidate_caches()
            print("[INFO]: retinalysis-inference installed.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR]: Failed to install retinalysis-inference: {e}", file=sys.stderr)

_ensure_rtnls_inference()


import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig
from datetime import timedelta

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(__file__).parent / "weights"
AV_WEIGHTS    = WEIGHTS_DIR / "av_july24.pt"
FOVEA_WEIGHTS = WEIGHTS_DIR / "fovea-july24.pt"
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

    # TorchScript .pt files load directly — no /tmp caching needed.
    # The module-level _get_runner() call keeps models in _runner across
    # warm invocations within the same container.
    logger.info(f"Loading VascX models from weights/ (PID={os.getpid()}) ...")

    class _Runner:
        seg        = SegmentationEnsemble.from_torchscript(str(AV_WEIGHTS))
        hm         = HeatmapRegressionEnsemble.from_torchscript(str(FOVEA_WEIGHTS))
        preprocess = staticmethod(parallel_preprocess)

    _runner = _Runner()

    logger.info("VascX models ready.")
    return _runner


# Models are loaded lazily on first tool call — do NOT pre-warm at import.
# Each .pt file is ~337 MB; loading both at startup exceeds the Lambda
# memory budget and causes a SIGKILL before the server comes up.
# _get_runner() caches the result in _runner so subsequent calls are free.

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

    try:
        await progress.set_total(3)
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

    try:
        await progress.set_total(3)
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

            y, x = np.unravel_index(np.argmax(hm_pred), hm_pred.shape)
            await progress.increment()
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
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
