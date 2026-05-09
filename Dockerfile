# =============================================================================
# fundus-mcp-vascx — Horizon MCP server
# =============================================================================
# Preprocessing (fundusprep) runs locally in this container.
# GPU inference is dispatched to a RunPod serverless endpoint at runtime —
# no model weights are baked into this image.
#
# Required environment variables (set in Horizon, not here):
#   RUNPOD_API_KEY       RunPod API key
#   RUNPOD_ENDPOINT_URL  Full RunPod endpoint base URL,
#                        e.g. https://api.runpod.ai/v2/<endpoint_id>
#   FASTMCP_DOCKET_URL   rediss://<host>:<port>  Redis for background tasks
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — Python dependency builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System build deps needed to compile some wheels (e.g. Pillow, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# ---------------------------------------------------------------------------
# Install order matters:
#
#   1. Pin opencv-python-headless FIRST so it is never overwritten by the
#      GUI variant that retinalysis-fundusprep pulls in transitively.
#   2. Install retinalysis-inference with --no-deps to bypass its tight
#      transitive pins that conflict with the rest of the stack.
#   3. Install everything else from requirements.txt normally.
# ---------------------------------------------------------------------------

# Step 1 — headless OpenCV, locked first
RUN pip install --no-cache-dir "opencv-python-headless>=4.0,<5.0"

# Step 2 — retinalysis-inference without dependency resolution
RUN pip install --no-cache-dir --no-deps retinalysis-inference

# Step 3 — everything else
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: confirm headless OpenCV won
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

# Minimal runtime libs needed by OpenCV headless + Pillow + libglib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully resolved site-packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source — no weights directory
COPY server.py .

# Non-root user
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# Horizon expects the process to listen on $PORT (default 8080)
EXPOSE 8080

CMD ["python", "server.py"]
