# Build arguments
ARG PYTHON_VERSION=3.10-slim

# Start Python image
FROM python:${PYTHON_VERSION}

# Install git and other dependencies
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install requirements
WORKDIR /app
COPY . .
RUN uv sync

# Setup virtual environment
ENV PATH="/app/.venv/bin:$PATH"