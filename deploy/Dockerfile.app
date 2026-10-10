# VibeRec web app. Built for the machine it runs on (arm64 on an M-series Mac and on AWS Graviton t4g instances).
FROM python:3.12.7-slim
WORKDIR /srv
COPY deploy/requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ src/
COPY app/ app/
ENV VIBEREC_DATA_DIR=/data AWS_REGION=us-east-2 PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
