"""
server.py — fundus-mcp-vascx
=============================
FastMCP server exposing VascX artery/vein segmentation and fovea
localization as MCP tools, deployed via Prefect Horizon.

Preprocessing runs locally (fundusprep); GPU inference is dispatched to a
Modal serverless endpoint so Horizon doesn't need a GPU or model weights.

Required environment variables:
    MODAL_ENDPOINT_URL  Full Modal endpoint base URL,
                        e.g. https://<workspace>--vascx-segmentation-fastapi-app.modal.run

Optional environment variables:
    FASTMCP_DOCKET_URL  rediss://<host>:<port>  Redis for background tasks

Tools:
    segment_av(image_b64, image_id)     → AV mask stats + base64 NPZ
    localize_fovea(image_b64, image_id) → fovea (x, y) coordinates
    health()                            → liveness check
"""

from __future__ import annotations

import subprocess
import sys


def _ensure_headless_opencv():
    try:
        import importlib.util
        spec = importlib.util.find_spec("cv2")
        if spec is None:
            return
        cv2_path = spec.origin or ""
        if "headless" not in cv2_path:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "--force-reinstall", "--no-deps", "opencv-python-headless",
            ])
            import importlib
            importlib.invalidate_caches()
    except Exception as e:
        print(f"[WARNING]: opencv headless check failed: {e}", file=sys.stderr)

_ensure_headless_opencv()


def _ensure_rtnls_inference():
    try:
        import rtnls_inference  # noqa: F401
    except ImportError:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet",
                "--no-deps", "retinalysis-inference",
            ])
            import importlib
            importlib.invalidate_caches()
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

import requests
from fastmcp import FastMCP

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODAL_ENDPOINT_URL = os.environ.get("MODAL_ENDPOINT_URL", "").rstrip("/")

if not MODAL_ENDPOINT_URL:
    logger.warning("MODAL_ENDPOINT_URL is not set — inference calls will fail.")


# ---------------------------------------------------------------------------
# Modal client
# ---------------------------------------------------------------------------

def _modal_dispatch(route: str, image_id: str, rgb_b64: str, ce_b64: str) -> dict:
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL is not set.")

    url = f"{MODAL_ENDPOINT_URL}{route}"
    logger.info(f"[{image_id}] Dispatching to Modal: {url}")

    resp = requests.post(
        url,
        json={"image_id": image_id, "rgb_b64": rgb_b64, "ce_b64": ce_b64},
        timeout=120,
    )
    resp.raise_for_status()
    output = resp.json()

    if not output.get("success"):
        raise RuntimeError(f"Modal inference failed: {output.get('error')}")

    return output


# ---------------------------------------------------------------------------
# Preprocessing — runs locally, no GPU needed
# ---------------------------------------------------------------------------

def _preprocess(image_b64: str, image_id: str, tmp: Path):
    import cv2
    from PIL import Image as _Image
    from rtnls_fundusprep.preprocessor import parallel_preprocess

    cv2.setNumThreads(1)

    rgb_dir  = tmp / "rgb"; rgb_dir.mkdir()
    ce_dir   = tmp / "ce";  ce_dir.mkdir()
    img_path = tmp / f"{image_id}.jpg"
    img_path.write_bytes(base64.b64decode(image_b64))

    bounds_list = parallel_preprocess(
        [img_path], rgb_path=rgb_dir, ce_path=ce_dir, n_jobs=1,
    )
    result = bounds_list[0]
    if not result.get("success", True):
        raise RuntimeError(f"Preprocessing failed: {result}")

    def _to_b64(p: Path) -> str:
        img = _Image.open(p).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    rgb_b64 = _to_b64(rgb_dir / f"{image_id}.png")
    ce_b64  = _to_b64(ce_dir  / f"{image_id}.png")
    return rgb_b64, ce_b64, result


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP("fundus-vascx")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def segment_av(image_b64: str, image_id: str) -> str:
    """
    Run VascX artery/vein segmentation on a fundus image.

    Preprocessing runs locally; GPU inference is dispatched to Modal.

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
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rgb_b64, ce_b64, bounds = _preprocess(image_b64, image_id, Path(tmp))
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})

        output = _modal_dispatch("/segment_av", image_id, rgb_b64, ce_b64)

        shape  = output["shape"]
        av_raw = np.frombuffer(
            base64.b64decode(output["av_raw_b64"]), dtype=np.uint8
        ).reshape(shape)

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
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
        logger.info(f"segment_av: {image_id}  payload={len(payload)/1024:.1f}KB")
        return payload

    except Exception as e:
        logger.error(f"segment_av failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def localize_fovea(image_b64: str, image_id: str) -> str:
    """
    Localise the fovea (macula centre) in a fundus image using VascX.

    Preprocessing runs locally; GPU inference is dispatched to Modal.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with fovea x, y coordinates in preprocessed image space.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                rgb_b64, ce_b64, _ = _preprocess(image_b64, image_id, Path(tmp))
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})

        output = _modal_dispatch("/localize_fovea", image_id, rgb_b64, ce_b64)

        return json.dumps({
            "success":  True,
            "image_id": image_id,
            "x":        output["x"],
            "y":        output["y"],
        })

    except Exception as e:
        logger.error(f"localize_fovea failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e), "image_id": image_id})


@mcp.tool()
async def health() -> str:
    """Liveness probe. Reports Modal endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "fundus-vascx",
        "modal": {
            "endpoint_url": MODAL_ENDPOINT_URL or "(not set)",
            "configured":   bool(MODAL_ENDPOINT_URL),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
