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

# Install Playwright system dependencies + Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libdbus-1-3 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
    libwayland-client0 fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install only Chromium (skip Firefox/WebKit to reduce image size)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install chromium

# Lambda needs a writable HOME for Chromium user data
ENV HOME=/tmp

# Copy application source
COPY app/ ${FUNCTION_DIR}/app/

ENTRYPOINT [ "/usr/bin/python3", "-m", "awslambdaric" ]
CMD [ "app.handler.lambda_handler" ]
