FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default entrypoint is the producer. Override with `--entrypoint python` for other modules.
ENTRYPOINT ["python", "-m", "streaming.producer"]
CMD ["--hours", "168"]
