# Dockerfile.ml - Multi-stage build
# ============================================================================
# IMPORTANT: Ansys Handling
# ============================================================================
# This Docker image is for the ML pipeline ONLY (data processing, training)
#
# Ansys simulation runs OUTSIDE the container because:
# 1. Ansys requires system license (cannot containerize)
# 2. Ansys is Windows/Linux specific (container is generic)
# 3. Ansys output is mounted as volume
#
# Workflow:
# 1. Run Ansys simulations on host/HPC cluster
# 2. Output files saved to ./data/simulations/
# 3. Docker container processes results
#
# For Student Ansys Edition:
# - Max 32,000 nodes per simulation
# - Single-threaded only
# - See README.md for deployment restrictions
# ============================================================================
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime (minimal)
FROM python:3.10-slim

# Build arguments (passed from CI)
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=unknown

# Labels for metadata
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.title="Polymer Informatics ML Pipeline"
LABEL org.opencontainers.image.description="ML pipeline for polymer capacitance prediction"
LABEL org.opencontainers.image.authors="Your Team"

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy only production code (exclude tests, docs)
COPY config/ config/
COPY data/ data/
COPY models/ models/
COPY simulation/ simulation/
COPY analysis/ analysis/
COPY utils/ utils/
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set PATH for user packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Print build info on startup (can be seen in docker logs or run command)
RUN echo "Build date: $BUILD_DATE" && \
    echo "Git commit: $VCS_REF" && \
    echo "Version: $VERSION"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from codes.config_manager import ConfigurationLoader; ConfigurationLoader.load_configuration()"

ENTRYPOINT ["python", "main.py"]
