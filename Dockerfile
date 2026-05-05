# Microsoft official Playwright Python image — Ubuntu 24.04 (noble) with
# Chromium + all system deps pre-installed and tested by the Playwright team.
# Tag MUST match playwright version pin in requirements.txt (1.56.0).
#
# Pattern verified from working production Lambda deployments:
#   - Stas Deep: stasdeep.com/articles/playwright-aws-lambda
#   - Verçosa:   github.com/lfvvercosa/example_playwright_lambda
# Both stages use the SAME base image. awslambdaric must be installed via
# pip --target so it lands in the function directory, not system site-packages.
ARG LAMBDA_TASK_ROOT="/var/task"

# Stage 1: Build — install awslambdaric (needs C++ toolchain)
FROM --platform=linux/amd64 mcr.microsoft.com/playwright/python:v1.56.0-noble AS build-image
ARG LAMBDA_TASK_ROOT
RUN mkdir -p ${LAMBDA_TASK_ROOT}
WORKDIR ${LAMBDA_TASK_ROOT}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ make cmake unzip libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --target ${LAMBDA_TASK_ROOT} awslambdaric

# Stage 2: Runtime — clean Microsoft base + copied artifacts
FROM --platform=linux/amd64 mcr.microsoft.com/playwright/python:v1.56.0-noble
ARG LAMBDA_TASK_ROOT
WORKDIR ${LAMBDA_TASK_ROOT}

# Bring awslambdaric over from the build stage
COPY --from=build-image ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

# Project Python deps (playwright already in base; pip will see it satisfied)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --target ${LAMBDA_TASK_ROOT} -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Lambda needs a writable HOME for Chromium user data
ENV HOME=/tmp

# Application source
COPY app/ ${LAMBDA_TASK_ROOT}/app/

# Microsoft's image installs python at /usr/bin/python — NOT /usr/local/bin/python3.
ENTRYPOINT [ "/usr/bin/python", "-m", "awslambdaric" ]
CMD [ "app.handler.lambda_handler" ]
