"""
server.py — fundus-mcp-vascx
=============================
FastMCP server exposing VascX artery/vein segmentation and fovea
localization as MCP tools, deployed via Prefect Horizon.

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Expected weight files:
    weights/artery-vein.pt    AV segmentation ensemble
    weights/vascx-fovea.pt    Fovea localization heatmap regression

Tools:
    segment_av(image_b64, image_id)     → AV mask stats + base64 NPZ
    localize_fovea(image_b64, image_id) → fovea (x, y) coordinates
    health()                            → liveness check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from pathlib import Path

from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("fundus-vascx")

WEIGHTS_DIR   = Path(__file__).parent / "weights"
AV_WEIGHTS    = WEIGHTS_DIR / "artery-vein.pt"
FOVEA_WEIGHTS = WEIGHTS_DIR / "vascx-fovea.pt"

# ---------------------------------------------------------------------------
# Lazy model loader — initialized once on first tool call
# ---------------------------------------------------------------------------

_runner = None


def _get_runner():
    global _runner
    if _runner is not None:
        return _runner

    import cv2
    cv2.setNumThreads(1)

    if not AV_WEIGHTS.exists():
        raise FileNotFoundError(f"AV weights not found: {AV_WEIGHTS}")
    if not FOVEA_WEIGHTS.exists():
        raise FileNotFoundError(f"Fovea weights not found: {FOVEA_WEIGHTS}")

    from rtnls_inference import SegmentationEnsemble, HeatmapRegressionEnsemble
    from rtnls_fundusprep.preprocessor import parallel_preprocess

    logger.info(f"Loading VascX models from {WEIGHTS_DIR} ...")

    class _Runner:
        seg        = SegmentationEnsemble.from_file(str(AV_WEIGHTS))
        hm         = HeatmapRegressionEnsemble.from_file(str(FOVEA_WEIGHTS))
        preprocess = staticmethod(parallel_preprocess)

    _runner = _Runner()
    logger.info("VascX models ready.")
    return _runner


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

@mcp.tool()
async def segment_av(image_b64: str, image_id: str) -> str:
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

    runner = _get_runner()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            rgb_img, ce_img, bounds = _preprocess(image_b64, image_id, tmp)
        except RuntimeError as e:
            return json.dumps({"success": False, "reason": str(e)})

        av_pred = runner.seg.predict_preprocessed(
            [np.array(rgb_img)], [np.array(ce_img)]
        )[0]

        av_raw = av_pred.astype(np.uint8)

        # Zero out pixels outside the fundus circle (black border = background)
        rgb_arr = np.array(rgb_img)
        av_raw[rgb_arr.max(axis=2) <= 10] = 0

        _A, _V, _X = 1, 2, 3
        artery = ((av_raw == _A) | (av_raw == _X)).astype(np.uint8)
        vein   = ((av_raw == _V) | (av_raw == _X)).astype(np.uint8)

        b  = bounds.get("bounds", {})
        cx = b.get("center", [0, 0])[0]
        cy = b.get("center", [0, 0])[1]
        r  = b.get("radius", 0)
        orig_h, orig_w = b.get("hw", (0, 0))

        buf = io.BytesIO()
        np.savez_compressed(buf, artery=artery, vein=vein, av_raw=av_raw)

        return json.dumps({
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


@mcp.tool()
async def localize_fovea(image_b64: str, image_id: str) -> str:
    """
    Localize the fovea (macula center) in a fundus image using VascX.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with fovea x, y coordinates in preprocessed image space.
    """
    import numpy as np

    runner = _get_runner()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            rgb_img, ce_img, _ = _preprocess(image_b64, image_id, tmp)
        except RuntimeError as e:
            return json.dumps({"success": False, "reason": str(e)})

        hm_pred = runner.hm.predict_preprocessed(
            [np.array(rgb_img)], [np.array(ce_img)]
        )[0]

        y, x = np.unravel_index(np.argmax(hm_pred), hm_pred.shape)

        return json.dumps({
            "success":  True,
            "image_id": image_id,
            "x":        float(x),
            "y":        float(y),
            "model":    FOVEA_WEIGHTS.name,
        })


@mcp.tool()
async def health() -> str:
    """Liveness probe."""
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
