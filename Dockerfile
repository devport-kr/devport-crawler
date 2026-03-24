# AWS Lambda Python 3.11 base image (Amazon Linux 2023)
FROM public.ecr.aws/lambda/python:3.11

# Install Python dependencies first — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source — this layer changes most often, so it goes last
COPY app/ ${LAMBDA_TASK_ROOT}/app/

CMD ["app.handler.lambda_handler"]
