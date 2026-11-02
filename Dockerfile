FROM python:3.12.15-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 INSPECTION_DL_DEVICE=cpu
COPY pyproject.toml requirements-cpu.txt requirements-cpu.lock ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-cpu.txt \
    && python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 inspection
USER inspection
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=3)"]
CMD ["python", "-m", "uvicorn", "inspection_dl.api:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "8"]
