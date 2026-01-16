#!/bin/bash

MAX_RETRIES=10
RETRY_COUNT=0

while true; do
    echo "Starting Python script... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
    source .venv/bin/activate

    uv run python src/eo_data/pipeline_data.py \
        --product "reflectance_250m" \
        --start_date "2025-06-01" \
        --end_date "2025-09-30" \
        --batch_days 5 \
        --store_cloud 

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Python script finished successfully."
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))

    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Reached maximum retries ($MAX_RETRIES). Exiting."
        exit $exit_code
    fi

    echo "Python script crashed with exit code $exit_code. Restarting in 2 seconds..."
    sleep 2
done






