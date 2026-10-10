#!/bin/bash
# Starts (or restarts) the database and the app on the server. Safe to run again after uploading a new image.
# Lives at /opt/viberec/start.sh (copied from S3 by update_server.sh). Settings come from /opt/viberec/env.
set -euo pipefail
source /opt/viberec/env
until aws s3 ls "s3://$VIBEREC_BUCKET/app/viberec-app.tar.gz" >/dev/null 2>&1; do sleep 30; done
aws s3 cp "s3://$VIBEREC_BUCKET/app/viberec-app.tar.gz" /tmp/app.tar.gz --only-show-errors && docker load -i /tmp/app.tar.gz && rm /tmp/app.tar.gz
docker network inspect viberec >/dev/null 2>&1 || docker network create viberec
if ! docker ps -a --format '{{.Names}}' | grep -q '^viberec-db$'; then
  docker run -d --name viberec-db --network viberec --restart unless-stopped \
    -e POSTGRES_USER=viberec -e POSTGRES_PASSWORD="$VIBEREC_DB_PASSWORD" -e POSTGRES_DB=viberec \
    -v viberec_pg:/var/lib/postgresql/data pgvector/pgvector:0.8.0-pg16 \
    -c shared_buffers=128MB -c work_mem=8MB -c max_connections=20
  until docker exec viberec-db pg_isready -U viberec >/dev/null 2>&1; do sleep 2; done
  if ! docker exec viberec-db psql -U viberec -d viberec -tAc "SELECT to_regclass('movies') IS NOT NULL" | grep -q t; then
    until aws s3 ls "s3://$VIBEREC_BUCKET/db/viberec.dump" >/dev/null 2>&1; do sleep 30; done
    aws s3 cp "s3://$VIBEREC_BUCKET/db/viberec.dump" /tmp/viberec.dump --only-show-errors
    docker cp /tmp/viberec.dump viberec-db:/tmp/viberec.dump
    docker exec viberec-db pg_restore -U viberec -d viberec --no-owner /tmp/viberec.dump
    rm /tmp/viberec.dump
  fi
fi
[ -f /opt/viberec/data/llm_spend_ledger.json ] || aws s3 cp "s3://$VIBEREC_BUCKET/data/llm_spend_ledger.json" /opt/viberec/data/ || true
docker rm -f viberec-app >/dev/null 2>&1 || true
docker run -d --name viberec-app --network viberec --restart unless-stopped -p 80:8000 \
  -e VIBEREC_DB_URL="postgresql://viberec:$VIBEREC_DB_PASSWORD@viberec-db:5432/viberec" \
  -e VIBEREC_API_KEY="${VIBEREC_API_KEY:-}" -e AWS_REGION="$AWS_REGION" -v /opt/viberec/data:/data viberec-app:latest \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
echo "VibeRec started"
