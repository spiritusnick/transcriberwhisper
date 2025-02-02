# Transcriber Whisper

A real-time audio transcription tool using OpenAI's Whisper, optimized for QuickTime Player input on macOS.

## Features
- Real-time transcription using Whisper's tiny.en model
- Automatic QuickTime Player input detection
- Saves transcripts to Desktop with timestamps in filename
- Optional timestamp display in transcripts
- Fast and optimized for real-time use

## Requirements
- Python 3.8 or higher
- ffmpeg
- portaudio
- Virtual environment with required packages

## Installation

1. Install system dependencies:
```bash
brew install ffmpeg portaudio
```

2. Set up Python environment:
```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
```

## Usage

1. Double-click `transcribe.command` to start transcribing
2. Output files are saved to Desktop as `transcript_YYYYMMDD_HHMMSS.txt`

### Command Line Options
- `--no-timestamps`: Disable timestamps in output
- `--model [tiny.en|base.en|small|medium|large]`: Choose Whisper model
- `--list-devices`: Show available audio devices
- `--input-device N`: Use specific input device
- `--output-file FILE`: Specify custom output file

## Files
- `live_transcribe.py`: Main transcription script
- `transcribe.command`: macOS launcher script
- `requirements.txt`: Python dependencies
