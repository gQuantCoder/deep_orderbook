#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/repos/gaelreinaudi/deep_orderbook
TRAIN=(
/media/photoDS216/crypto/2025-04-09T00-00-11.parquet
/media/photoDS216/crypto/2025-04-09T01-16-15.parquet
/media/photoDS216/crypto/2025-04-09T02-00-32.parquet
/media/photoDS216/crypto/2025-04-09T03-00-32.parquet
/media/photoDS216/crypto/2025-04-09T04-00-11.parquet
/media/photoDS216/crypto/2025-04-09T05-00-12.parquet
/media/photoDS216/crypto/2025-04-09T06-00-11.parquet
/media/photoDS216/crypto/2025-04-09T07-00-11.parquet
/media/photoDS216/crypto/2025-04-09T08-00-32.parquet
/media/photoDS216/crypto/2025-04-09T09-00-11.parquet
/media/photoDS216/crypto/2025-04-09T10-00-11.parquet
/media/photoDS216/crypto/2025-04-09T11-00-17.parquet
/media/photoDS216/crypto/2025-04-09T12-00-12.parquet
/media/photoDS216/crypto/2025-04-09T13-00-32.parquet
/media/photoDS216/crypto/2025-04-09T14-00-32.parquet
/media/photoDS216/crypto/2025-04-09T15-00-11.parquet
)
VAL=(
/media/photoDS216/crypto/2025-04-09T16-00-12.parquet
/media/photoDS216/crypto/2025-04-09T17-00-32.parquet
/media/photoDS216/crypto/2025-04-09T18-00-33.parquet
/media/photoDS216/crypto/2025-04-09T19-00-33.parquet
)
mkdir -p /tmp/exp34_logs
for TEST in "${VAL[@]}"; do
  base=$(basename "$TEST" .parquet)
  label="exp34_val_${base}"
  log="/tmp/exp34_logs/${label}.log"
  echo "RUN $label"
  ./.venv/bin/python scripts/exp22_trigger_sweep.py \
    --label "$label" \
    --variants l1_evt005_pw2_h64 \
    --train-files "${TRAIN[@]}" \
    --test-files "$TEST" \
    --directions long short > "$log" 2>&1
  tail -n 5 "$log"
done
