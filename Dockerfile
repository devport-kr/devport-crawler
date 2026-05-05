# Stage 1: Build awslambdaric (AWS Lambda Runtime Interface Client)
ARG FUNCTION_DIR="/var/task"
FROM --platform=linux/arm64 python:3.11-slim-bookworm AS build-image
ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

RUN apt-get update && apt-get install -y g++ make cmake unzip libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --target ${FUNCTION_DIR} awslambdaric

# Stage 2: Runtime
FROM --platform=linux/arm64 python:3.11-slim-bookworm
ARG FUNCTION_DIR="/var/task"
WORKDIR ${FUNCTION_DIR}

# Copy awslambdaric from build stage
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Install Python dependencies first so playwright is available for install-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Let Playwright install its own canonical system dependency list.
# This avoids drift between hand-maintained apt list and what Chromium actually
# needs on Debian 12 arm64 (libxshmfence1, libxext6, libxrender1, libxtst6,
# libxss1, libx11-xcb1, libxcb-dri3-0, libgles2 etc).
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Lambda's sbx_user must be able to read browser binaries
RUN chmod -R o+rx /ms-playwright

# Lambda needs a writable HOME for Chromium user data
ENV HOME=/tmp

# Copy application source
COPY app/ ${FUNCTION_DIR}/app/

ENTRYPOINT [ "/usr/local/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.handler.lambda_handler" ]
