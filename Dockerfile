# Stage 1: Build awslambdaric (AWS Lambda Runtime Interface Client)
ARG FUNCTION_DIR="/var/task"
FROM mcr.microsoft.com/playwright/python:v1.52.0-noble AS build-image
ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

RUN apt-get update && apt-get install -y g++ make cmake unzip libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --target ${FUNCTION_DIR} awslambdaric

# Stage 2: Runtime
FROM mcr.microsoft.com/playwright/python:v1.52.0-noble
ARG FUNCTION_DIR="/var/task"
WORKDIR ${FUNCTION_DIR}

# Copy awslambdaric from build stage
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

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

ENTRYPOINT [ "/usr/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler.lambda_handler" ]
