"""
server.py — fundus-mcp-vascx
=============================
FastMCP server exposing VascX artery/vein segmentation and fovea
localisation as MCP tools, for deployment on Prefect Horizon.

Weights are committed to the repo via Git LFS and loaded from
the ./weights/ directory at startup. No download required.

Tools:
    segment_av(image_b64, image_id)     → AV mask stats + base64 NPZ
    localise_fovea(image_b64, image_id) → fovea (x, y) coordinates
    health()                            → liveness check
"""

from __future__ import annotations

import asyncio
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

# Weights are in ./weights/ relative to this file
WEIGHTS_DIR = Path(__file__).parent / "weights"

# ---------------------------------------------------------------------------
# Lazy model loader — initialised once on first tool call
# ---------------------------------------------------------------------------

_runner = None


def _get_runner():
    global _runner
    if _runner is not None:
        return _runner

    import cv2
    cv2.setNumThreads(1)

    os.environ["HF_HOME"] = str(WEIGHTS_DIR)

    from rtnls_inference import SegmentationEnsemble, HeatmapRegressionEnsemble
    from rtnls_fundusprep.preprocessor import parallel_preprocess

    logger.info(f"Loading VascX models from {WEIGHTS_DIR} ...")

    class _Runner:
        seg       = SegmentationEnsemble.from_name("artery_vein/av_july24")
        hm        = HeatmapRegressionEnsemble.from_name("fovea/fovea_oct24")
        preprocess = staticmethod(parallel_preprocess)

    _runner = _Runner()
    logger.info("VascX models ready.")
    return _runner


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
    import cv2
    import numpy as np
    from PIL import Image as _Image
    from datetime import datetime, timezone

    cv2.setNumThreads(1)
    runner = _get_runner()

    with tempfile.TemporaryDirectory() as tmp:
        tmp      = Path(tmp)
        img_path = tmp / f"{image_id}.jpg"
        rgb_dir  = tmp / "rgb";  rgb_dir.mkdir()
        ce_dir   = tmp / "ce";   ce_dir.mkdir()

        img_path.write_bytes(base64.b64decode(image_b64))

        bounds_list = runner.preprocess(
            [img_path], rgb_path=rgb_dir, ce_path=ce_dir, n_jobs=1,
        )
        result = bounds_list[0]
        if not result.get("success", True):
            return json.dumps({"success": False,
                               "reason": f"Preprocessing failed: {result}"})

        rgb_img = _Image.open(rgb_dir / f"{image_id}.png").convert("RGB")
        ce_img  = _Image.open(ce_dir  / f"{image_id}.png").convert("RGB")

        av_pred = runner.seg.predict_preprocessed(
            [np.array(rgb_img)], [np.array(ce_img)]
        )[0]

        av_raw     = av_pred.astype(np.uint8)
        rgb_arr    = np.array(rgb_img)
        fundus_ok  = rgb_arr.max(axis=2) > 10
        av_raw[~fundus_ok] = 0

        _A, _V, _X = 1, 2, 3
        artery = ((av_raw == _A) | (av_raw == _X)).astype(np.uint8)
        vein   = ((av_raw == _V) | (av_raw == _X)).astype(np.uint8)

        b  = result.get("bounds", {})
        cx = b.get("center", [0, 0])[0]
        cy = b.get("center", [0, 0])[1]
        r  = b.get("radius", 0)
        orig_h, orig_w = b.get("hw", (0, 0))

        buf = io.BytesIO()
        np.savez_compressed(buf, artery=artery, vein=vein, av_raw=av_raw)

        return json.dumps({
            "success":             True,
            "image_id":            image_id,
            "shape":               list(av_raw.shape),
            "artery_pixel_count":  int(artery.sum()),
            "vein_pixel_count":    int(vein.sum()),
            "crossing_pixel_count":int((av_raw == _X).sum()),
            "vessel_pixel_count":  int((av_raw >= 1).sum()),
            "crop_bounds": {
                "cx": cx, "cy": cy, "radius": r,
                "orig_h": orig_h, "orig_w": orig_w,
                "x1": max(0, cx - r), "y1": max(0, cy - r),
                "x2": min(orig_w, cx + r), "y2": min(orig_h, cy + r),
            },
            "masks_b64": base64.b64encode(buf.getvalue()).decode(),
            "model":     "Eyened/vascx:artery_vein/av_july24",
            "created_at": datetime.utcnow().isoformat() + "Z",
        })


@mcp.tool()
async def localise_fovea(image_b64: str, image_id: str) -> str:
    """
    Localise the fovea (macula centre) in a fundus image using VascX.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with fovea x, y coordinates in preprocessed image space.
    """
    import cv2
    import numpy as np
    from PIL import Image as _Image

    cv2.setNumThreads(1)
    runner = _get_runner()

    with tempfile.TemporaryDirectory() as tmp:
        tmp      = Path(tmp)
        img_path = tmp / f"{image_id}.jpg"
        rgb_dir  = tmp / "rgb";  rgb_dir.mkdir()
        ce_dir   = tmp / "ce";   ce_dir.mkdir()

        img_path.write_bytes(base64.b64decode(image_b64))

        bounds_list = runner.preprocess(
            [img_path], rgb_path=rgb_dir, ce_path=ce_dir, n_jobs=1,
        )
        result = bounds_list[0]
        if not result.get("success", True):
            return json.dumps({"success": False,
                               "reason": f"Preprocessing failed: {result}"})

        rgb_img = _Image.open(rgb_dir / f"{image_id}.png").convert("RGB")
        ce_img  = _Image.open(ce_dir  / f"{image_id}.png").convert("RGB")

        hm_pred = runner.hm.predict_preprocessed(
            [np.array(rgb_img)], [np.array(ce_img)]
        )[0]

        y, x = np.unravel_index(np.argmax(hm_pred), hm_pred.shape)

        return json.dumps({
            "success":  True,
            "image_id": image_id,
            "x":        float(x),
            "y":        float(y),
        })


@mcp.tool()
async def health() -> str:
    """Liveness probe."""
    return json.dumps({"status": "ok", "service": "fundus-vascx",
                       "weights_dir": str(WEIGHTS_DIR)})
