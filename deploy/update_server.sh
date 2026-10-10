#!/bin/bash
# One-time update to app v2 (run on the server through Session Manager:  sudo bash /tmp/update_server.sh).
# - installs the new start script, the API key and the nightly backup timer;
# - replaces the database with the new dump ONLY if it holds no user profiles yet (refuses otherwise);
# - restarts everything.
set -euo pipefail
source /opt/viberec/env
B="s3://$VIBEREC_BUCKET/deploy"
aws s3 cp "$B/start.sh" /opt/viberec/start.sh --only-show-errors && chmod 700 /opt/viberec/start.sh
aws s3 cp "$B/backup.sh" /opt/viberec/backup.sh --only-show-errors && chmod 700 /opt/viberec/backup.sh
KEY="$(aws s3 cp "$B/api_key" - 2>/dev/null || true)"
if [ -n "$KEY" ] && ! grep -q '^VIBEREC_API_KEY=' /opt/viberec/env; then echo "VIBEREC_API_KEY=$KEY" >> /opt/viberec/env; fi
cat > /etc/systemd/system/viberec-backup.service <<'UNIT'
[Unit]
Description=VibeRec nightly database backup to S3
[Service]
Type=oneshot
ExecStart=/opt/viberec/backup.sh
UNIT
cat > /etc/systemd/system/viberec-backup.timer <<'UNIT'
[Unit]
Description=Run the VibeRec backup every night at 03:00 UTC
[Timer]
OnCalendar=*-*-* 03:00:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload && systemctl enable --now viberec-backup.timer
if docker exec viberec-db psql -U viberec -d viberec -tAc "SELECT to_regclass('profiles') IS NOT NULL" | grep -q t && \
   docker exec viberec-db psql -U viberec -d viberec -tAc "SELECT EXISTS (SELECT 1 FROM profiles)" | grep -q t; then
  echo "The database already holds user profiles: not replacing it. Only the app is restarted."
else
  docker rm -f viberec-app viberec-db >/dev/null 2>&1 || true
  docker volume rm viberec_pg >/dev/null 2>&1 || true
fi
/opt/viberec/start.sh
systemctl list-timers viberec-backup.timer --no-pager | head -3
