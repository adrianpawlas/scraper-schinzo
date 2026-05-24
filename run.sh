#!/usr/bin/env zsh
# Scheduled runner for scraper-schinzo
# Called by cron or manually: bash run.sh

set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/scraper-$TIMESTAMP.log"

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

echo "[$(date)] Starting scraper..." | tee -a "$LOG_FILE"

python3 scraper.py >> "$LOG_FILE" 2>&1

echo "[$(date)] Finished (exit code: $?)" | tee -a "$LOG_FILE"
