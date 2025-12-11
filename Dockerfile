FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright browsers (크롤링에서 필요할 경우 사용)
RUN pip install playwright==1.41.0 && \
    playwright install chromium && \
    playwright install-deps chromium

COPY app/ ${LAMBDA_TASK_ROOT}/app/

CMD ["app.handler.lambda_handler"]
