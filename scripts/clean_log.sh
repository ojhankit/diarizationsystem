#!/bin/bash
LOG_FILE="../diarization.log"
if [ -f "$LOG_FILE" ]; then
    truncate -s 0 "$LOG_FILE"
fi
