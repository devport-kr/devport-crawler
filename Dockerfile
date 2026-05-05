# Microsoft official Playwright Python image — Ubuntu 24.04 (noble) with
# Chromium + all system deps pre-installed and tested by the Playwright team.
# This is the only base image setup that has multiple confirmed working
# production Lambda deployments. Our previous python:3.11-slim-bookworm path
# kept hitting Chromium child-process issues that --single-process workarounds
# couldn't reliably fix.
#
# Tag MUST match playwright version pin in requirements.txt (1.56.0).
ARG FUNCTION_DIR="/var/task"

# Stage 1: Build awslambdaric on a clean Python image
FROM --platform=linux/arm64 python:3.11-slim-bookworm AS build-image
ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

RUN apt-get update && apt-get install -y g++ make cmake unzip libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --target ${FUNCTION_DIR} awslambdaric


# Stage 2: Runtime — Microsoft Playwright base
FROM --platform=linux/arm64 mcr.microsoft.com/playwright/python:v1.56.0-noble
ARG FUNCTION_DIR="/var/task"
WORKDIR ${FUNCTION_DIR}

# Copy awslambdaric from build stage
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Install Python dependencies (playwright already pre-installed in base image,
# but pip will see it satisfied and skip re-install).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Browser binaries are at the default Microsoft path; do NOT override
# PLAYWRIGHT_BROWSERS_PATH or playwright won't find them.

# Lambda needs a writable HOME for Chromium user data
ENV HOME=/tmp

# Copy application source
COPY app/ ${FUNCTION_DIR}/app/

ENTRYPOINT [ "/usr/local/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.handler.lambda_handler" ]
