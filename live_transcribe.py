import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import sys
import argparse
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define default devices
QUICKTIME_INPUT = 6  # QuickTime Player Input
SCREEN_RECORD_OUTPUT = 7  # screen record output

def find_device_by_name(name_contains):
    """Find a device ID by partial name match"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_contains.lower() in device['name'].lower():
            return i
    return None

def list_audio_devices():
    """List all available audio devices"""
    print("\nAvailable audio devices:")
    print("------------------------")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Max Channels: Input={device['max_input_channels']}, Output={device['max_output_channels']}")
        print(f"   Default Sample Rate: {device['default_samplerate']}Hz")
        print()
    return devices

def parse_arguments():
    parser = argparse.ArgumentParser(description='Live audio transcription with Whisper')
    parser.add_argument('--input-file', type=str, help='Input audio file (if not specified, uses QuickTime Player Input)')
    parser.add_argument('--output-file', type=str, help='Output file to save transcriptions')
    parser.add_argument('--no-timestamps', action='store_true', help='Disable timestamps in output')
    parser.add_argument('--model', default='tiny.en', choices=['tiny.en', 'base.en', 'tiny', 'base', 'small', 'medium', 'large'],
                      help='Whisper model to use (default: tiny.en)')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices and exit')
    parser.add_argument('--input-device', type=int, help='Override default QuickTime Player Input device')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate (default: 16000)')
    return parser.parse_args()

# Parse arguments first
args = parse_arguments()

# List devices if requested
if args.list_devices:
    list_audio_devices()
    sys.exit(0)

# Find QuickTime input device if not specified
if args.input_device is None:
    quicktime_id = find_device_by_name("Quicktime Player Input")
    if quicktime_id is not None:
        args.input_device = quicktime_id
    else:
        args.input_device = QUICKTIME_INPUT  # Fallback to default ID

# Initialize Whisper model
print(f"Loading Whisper model ({args.model})...")
model = whisper.load_model(args.model)
print("Model loaded!")

# Audio parameters
SAMPLE_RATE = args.sample_rate
CHANNELS = 1
CHUNK_DURATION = 1.5  # seconds (reduced from 3 to 1.5)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_SIZE = 4096  # Smaller buffer size for faster processing

# Create a queue to store audio chunks
audio_queue = queue.Queue()

# Flag to control recording
is_recording = True

def write_output(text, timestamp=None):
    """Write output to both console and file if specified"""
    if not text.strip():
        return
        
    if timestamp and not args.no_timestamps:
        output_line = f"[{timestamp}] {text}"
    else:
        output_line = text
    
    print(output_line, flush=True)
    
    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write(output_line + '\n')
            f.flush()

def audio_callback(indata, frames, time, status):
    """Callback function to process audio input"""
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def process_audio():
    """Process audio chunks from the queue"""
    while is_recording:
        try:
            # Collect audio for CHUNK_DURATION seconds
            audio_data = []
            start_time = time.time()
            
            while time.time() - start_time < CHUNK_DURATION:
                if not audio_queue.empty():
                    audio_data.append(audio_queue.get())
                else:
                    time.sleep(0.001)
            
            if audio_data:
                # Concatenate all chunks
                audio_chunk = np.concatenate(audio_data)
                
                # Ensure audio is in the correct format (mono, float32)
                if len(audio_chunk.shape) > 1:
                    audio_chunk = audio_chunk.mean(axis=1)
                
                # Normalize audio
                audio_chunk = audio_chunk.astype(np.float32)
                
                # Transcribe with optimized settings
                result = model.transcribe(
                    audio_chunk,
                    language='en',
                    task='transcribe',
                    fp16=False,
                    without_timestamps=True
                )
                
                # Output result if there's actual speech
                if result["text"].strip():
                    timestamp = datetime.now().strftime("%H:%M:%S") if not args.no_timestamps else None
                    write_output(result["text"].strip(), timestamp)
                
        except Exception as e:
            print(f"Error processing audio: {e}")

def process_file(filename):
    """Process an audio file instead of microphone input"""
    try:
        print(f"Transcribing file: {filename}")
        result = model.transcribe(filename, language='en', task='transcribe', fp16=False)
        
        # Output the transcription
        if args.no_timestamps:
            write_output(result["text"].strip())
        else:
            # For files, we'll use segment-level timestamps
            for segment in result["segments"]:
                timestamp = time.strftime("%H:%M:%S", time.gmtime(segment["start"]))
                write_output(segment["text"].strip(), timestamp)
                
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    """Main function to run the transcription"""
    global is_recording
    
    # If input file is specified, process it and exit
    if args.input_file:
        process_file(args.input_file)
        return
    
    # Otherwise, start live transcription
    print("Starting live transcription... (Press Ctrl+C to stop)")
    
    # Clear output file if it exists
    if args.output_file:
        open(args.output_file, 'w').close()
    
    # Start the processing thread
    processing_thread = threading.Thread(target=process_audio)
    processing_thread.start()
    
    try:
        # Start recording with specified device if provided
        stream_kwargs = {
            'samplerate': SAMPLE_RATE,
            'channels': CHANNELS,
            'callback': audio_callback,
            'blocksize': BUFFER_SIZE
        }
        if args.input_device is not None:
            stream_kwargs['device'] = args.input_device
        
        with sd.InputStream(**stream_kwargs):
            print(f"Using input device: {sd.query_devices(args.input_device)['name'] if args.input_device is not None else 'default'}")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        is_recording = False
        processing_thread.join()
        print("Transcription stopped.")
    except Exception as e:
        print(f"Error: {e}")
        is_recording = False
        processing_thread.join()

if __name__ == "__main__":
    main() 