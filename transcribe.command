#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a timestamp for the output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${HOME}/Desktop/transcript_${TIMESTAMP}.txt"

# Activate the virtual environment
source "${SCRIPT_DIR}/whisper-env/bin/activate"

# Run the transcription script with optimized settings
echo "Starting transcription..."
echo "Output will be saved to: ${OUTPUT_FILE}"
python "${SCRIPT_DIR}/live_transcribe.py" \
    --output-file "${OUTPUT_FILE}" \
    --model tiny.en \
    --no-timestamps \
    "$@"

# Keep the terminal window open
read -p "Press Enter to close..." 