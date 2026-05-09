"""
server.py — fundus-mcp-vascx
=============================
FastMCP server exposing VascX artery/vein segmentation and fovea
localization as MCP tools, deployed via Prefect Horizon.

Preprocessing runs locally (fundusprep); GPU inference is dispatched to a
RunPod serverless endpoint so Horizon doesn't need a GPU or model weights.

Required environment variables:
    RUNPOD_API_KEY       RunPod API key
    RUNPOD_ENDPOINT_URL  Full RunPod endpoint base URL,
                         e.g. https://api.runpod.ai/v2/<endpoint_id>

Optional environment variables:
    FASTMCP_DOCKET_URL   rediss://<host>:<port>  Redis for background tasks
    RUNPOD_POLL_INTERVAL Seconds between status polls (default: 3)
    RUNPOD_MAX_WAIT      Seconds before timeout (default: 120)

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
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Ensure retinalysis-inference is importable.
# ---------------------------------------------------------------------------
def _ensure_rtnls_inference():
    try:
        import rtnls_inference  # noqa: F401
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
import time
import tempfile
from pathlib import Path

import requests
from fastmcp import FastMCP
from fastmcp.dependencies import Progress
from fastmcp.server.tasks import TaskConfig
from datetime import timedelta

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNPOD_API_KEY       = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_URL  = os.environ.get("RUNPOD_ENDPOINT_URL", "").rstrip("/")
RUNPOD_POLL_INTERVAL = int(os.environ.get("RUNPOD_POLL_INTERVAL", "3"))
RUNPOD_MAX_WAIT      = int(os.environ.get("RUNPOD_MAX_WAIT", "120"))

if not RUNPOD_API_KEY:
    logger.warning("RUNPOD_API_KEY is not set — inference calls will fail.")
if not RUNPOD_ENDPOINT_URL:
    logger.warning("RUNPOD_ENDPOINT_URL is not set — inference calls will fail.")

_redis_url = os.environ.get("FASTMCP_DOCKET_URL")
if not _redis_url:
    logger.warning(
        "FASTMCP_DOCKET_URL is not set — background tasks will fail across "
        "stateless HTTP requests. Set this to your Redis URL in Horizon env vars."
    )


# ---------------------------------------------------------------------------
# RunPod client
# ---------------------------------------------------------------------------

def _runpod_session() -> requests.Session:
    """Return a requests Session pre-configured with RunPod auth headers."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    })
    return session


def _runpod_dispatch(task: str, image_id: str, rgb_b64: str, ce_b64: str) -> dict:
    """
    Submit a job to the RunPod serverless endpoint and poll until complete.

    Returns the output dict on success. Raises RuntimeError on failure or timeout.
    """
    session = _runpod_session()

    resp = session.post(
        f"{RUNPOD_ENDPOINT_URL}/run",
        json={"input": {
            "task":     task,
            "image_id": image_id,
            "rgb_b64":  rgb_b64,
            "ce_b64":   ce_b64,
        }},
    )
    resp.raise_for_status()
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"No job ID in RunPod response: {resp.json()}")
    logger.info(f"[{image_id}] RunPod job submitted: {job_id}")

    deadline = time.time() + RUNPOD_MAX_WAIT
    while time.time() < deadline:
        resp = session.get(f"{RUNPOD_ENDPOINT_URL}/status/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        logger.info(f"[{image_id}] RunPod status: {status}")
        if status == "COMPLETED":
            output = data.get("output", {})
            if not output.get("success"):
                raise RuntimeError(f"RunPod job completed but reported failure: {output.get('error')}")
            return output
        if status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error')}")
        time.sleep(RUNPOD_POLL_INTERVAL)

    raise TimeoutError(f"RunPod job {job_id} did not complete within {RUNPOD_MAX_WAIT}s")


# ---------------------------------------------------------------------------
# Preprocessing — runs locally on Horizon, no GPU needed
# ---------------------------------------------------------------------------

def _preprocess(image_b64: str, image_id: str, tmp: Path):
    """
    Decode image, run parallel_preprocess, return (rgb_b64, ce_b64, bounds).

    Returns base64-encoded PNG strings ready to POST directly to RunPod.
    """
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

@mcp.tool(task=TaskConfig(mode="required", poll_interval=timedelta(seconds=5)))
async def segment_av(
    image_b64: str,
    image_id: str,
    progress: Progress = Progress(),
) -> str:
    """
    Run VascX artery/vein segmentation on a fundus image.

    Preprocessing runs locally; GPU inference is dispatched to RunPod.

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
        await progress.set_message("Preprocessing image...")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                rgb_b64, ce_b64, bounds = _preprocess(image_b64, image_id, Path(tmp))
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})

        await progress.increment()
        await progress.set_message("Running AV segmentation on RunPod...")

        output = _runpod_dispatch("av", image_id, rgb_b64, ce_b64)

        await progress.increment()
        await progress.set_message("Computing mask statistics...")

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

    Preprocessing runs locally; GPU inference is dispatched to RunPod.

    Args:
        image_b64:  Base64-encoded RGB fundus image (JPEG or PNG).
        image_id:   Identifier for this image.

    Returns:
        JSON with fovea x, y coordinates in preprocessed image space.
    """
    try:
        await progress.set_total(3)
        await progress.set_message("Preprocessing image...")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                rgb_b64, ce_b64, _ = _preprocess(image_b64, image_id, Path(tmp))
            except RuntimeError as e:
                return json.dumps({"success": False, "reason": str(e)})

        await progress.increment()
        await progress.set_message("Localising fovea on RunPod...")

        output = _runpod_dispatch("fovea", image_id, rgb_b64, ce_b64)

        await progress.increment()
        await progress.set_message("Done.")
        await progress.increment()

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
    """Liveness probe. Reports RunPod endpoint configuration status."""
    return json.dumps({
        "status":  "ok",
        "service": "fundus-vascx",
        "runpod": {
            "endpoint_url":    RUNPOD_ENDPOINT_URL or "(not set)",
            "api_key_present": bool(RUNPOD_API_KEY),
            "configured":      bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_URL),
        },
    })


if __name__ == "__main__":
    mcp.run(stateless_http=True, json_response=True)
