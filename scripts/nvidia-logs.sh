#!/bin/bash

set -e

COMMAND=$1
LOG_FILE=$2
INTERVAL=${3:-15}

if [ "$COMMAND" == "monitor" ]; then
    echo "Starting GPU monitoring every ${INTERVAL}s into $LOG_FILE"
    while true; do
        timestamp=$(date --iso-8601=seconds)
        nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.total,memory.used,memory.free \
                   --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util mem_total mem_used mem_free; do
            echo "{\"timestamp\": \"$timestamp\", \"gpu_index\": $idx, \"name\": \"$name\", \"utilization\": $util, \"mem_total\": $mem_total, \"mem_used\": $mem_used, \"mem_free\": $mem_free}" >> "$LOG_FILE"
        done
        sleep "$INTERVAL"
    done

elif [ "$COMMAND" == "parse" ]; then
    echo "Summarizing GPU usage from $LOG_FILE"
    python3 - <<EOF
import json
from collections import defaultdict
import sys

logfile = "$LOG_FILE"
data = defaultdict(list)

with open(logfile) as f:
    for line in f:
        entry = json.loads(line)
        gpu = entry['gpu_index']
        data[gpu].append(entry)

for gpu, records in data.items():
    utils = [r["utilization"] for r in records]
    print(f"GPU {gpu} — Entries: {len(utils)}, Avg Utilization: {sum(utils)/len(utils):.2f}%")
EOF

else
    echo "Usage:"
    echo "  $0 monitor <log_file> [interval_seconds]"
    echo "  $0 parse <log_file>"
    exit 1
fi
