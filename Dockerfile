# =============================================================================
# fundus-mcp-vascx — Horizon MCP server
# =============================================================================
# Build args
#   WEIGHTS_DIR  local path (or Git LFS export) containing av_july24.pt and
#                fovea-july24.pt — baked into the image so Horizon has them
#                without a runtime download.
#
# Environment variables (set in Horizon, not here)
#   FASTMCP_DOCKET_URL   rediss://<host>:<port>  Redis for background tasks
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Python dependency builder
# Isolates all pip work so the final image only copies what's needed.
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System build deps needed to compile some wheels (e.g. Pillow, numpy, MONAI)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# ---------------------------------------------------------------------------
# Install order matters here — this is the same problem server.py works around
# at runtime. We solve it properly at build time:
#
#   1. Pin opencv-python-headless FIRST so it is never overwritten by the
#      GUI variant that retinalysis-fundusprep/vascx pulls in transitively.
#   2. Install retinalysis-inference with --no-deps to bypass its tight
#      transitive pins (huggingface-hub==0.25.1, albumentations==1.3.1 etc.)
#      that conflict with the rest of the stack.
#   3. Install everything else from requirements.txt normally.
#
# By doing this at build time, the two runtime self-healing blocks in
# server.py (_ensure_headless_opencv / _ensure_rtnls_inference) become
# no-ops on every warm invocation — no subprocess overhead, no pip calls.
# ---------------------------------------------------------------------------

# Step 1 — headless OpenCV, locked first
RUN pip install --no-cache-dir "opencv-python-headless>=4.0,<5.0"

# Step 2 — retinalysis-inference without dependency resolution
#           (its pins fight the rest of the stack; we own the resolution here)
RUN pip install --no-cache-dir --no-deps retinalysis-inference

# Step 3 — everything else; opencv-python-headless is already satisfied so
#           pip won't pull in the GUI variant as a transitive dep
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: confirm headless build won. If the GUI variant somehow won,
# cv2.__file__ will contain "headless"; the GUI variant path will not.
RUN python - <<'EOF'
import cv2, sys
path = cv2.__file__ or ""
print(f"cv2 path: {path}")
if "headless" not in path:
    sys.exit("ERROR: opencv-python (GUI) variant won — image will fail on Horizon")
print("OK: opencv-python-headless confirmed")
EOF


# -----------------------------------------------------------------------------
# Stage 2 — runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Minimal runtime libs needed by OpenCV headless + Pillow + libglib (MONAI/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully resolved site-packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source
COPY server.py .

# ---------------------------------------------------------------------------
# Weights — baked into the image.
#
# IMPORTANT: the files in the repo are Git LFS pointers (~130 B text files),
# not the actual model weights. Before building, ensure you have run:
#
#   git lfs pull
#
# in the repo root so the real .pt files (~337 MB each) are present locally.
# The COPY below will fail fast with a clear error if only the LFS pointer
# files are present (they are ~130 bytes; torch.load will raise immediately).
#
# Alternative: if weights are managed externally (e.g. GCS, S3, Weights &
# Biases), remove the COPY and mount them as a volume or download at startup
# using a separate entrypoint script. See comments in WEIGHTS_STRATEGY below.
# ---------------------------------------------------------------------------
COPY weights/av_july24.pt    weights/av_july24.pt
COPY weights/fovea-july24.pt weights/fovea-july24.pt

# Validate weight files at build time — catch LFS pointer files early.
# A real TorchScript .pt is a ZIP archive; LFS pointer text files are not.
RUN python - <<'EOF'
import zipfile, sys
for path in ["weights/av_july24.pt", "weights/fovea-july24.pt"]:
    try:
        with zipfile.ZipFile(path) as _:
            pass
        print(f"OK: {path} is a valid archive")
    except zipfile.BadZipFile:
        sys.exit(
            f"ERROR: {path} appears to be a Git LFS pointer, not the real weights.\n"
            "Run `git lfs pull` before building the image."
        )
EOF

# Non-root user — good hygiene for container deployments
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# Horizon expects the process to listen on $PORT (default 8080).
# FastMCP's stateless_http transport binds to 0.0.0.0:$PORT automatically
# when the PORT env var is set; we expose 8080 as the conventional default.
EXPOSE 8080

# No ENTRYPOINT wrapper needed — server.py calls mcp.run() in __main__.
# Horizon injects PORT; FastMCP reads it from the environment.
CMD ["python", "server.py"]
#   Mount a PVC or host path to /app/weights at runtime. Zero cold start,
#   zero image bloat. Not applicable to Horizon's serverless model.
# =============================================================================
