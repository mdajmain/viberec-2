#!/bin/bash
# Nightly database backup to S3 (run by the viberec-backup systemd timer). Old backups expire by the bucket's rule.
set -euo pipefail
source /opt/viberec/env
f="/tmp/viberec-$(date -u +%Y-%m-%d).dump"
docker exec viberec-db pg_dump -U viberec -Fc viberec > "$f"
aws s3 cp "$f" "s3://$VIBEREC_BUCKET/backups/$(basename "$f")" --only-show-errors
aws s3 cp /opt/viberec/data/llm_spend_ledger.json "s3://$VIBEREC_BUCKET/backups/llm_spend_ledger.json" --only-show-errors || true
rm -f "$f"
echo "backup done: $(basename "$f")"
