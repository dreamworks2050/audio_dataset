# Audio Transcription Module

This module provides functionality to transcribe audio files using whisper.cpp with the large-v3-turbo model.

## Setup

Before using the transcription functionality, you need to set up whisper.cpp and download the model:

```bash
# Run the setup script
./setup_whisper.py
```

This will:
1. Clone the whisper.cpp repository
2. Compile whisper.cpp
3. Download the large-v3-turbo model

## Usage

1. First, split your audio files using the "Split Audio" tab
2. Then, go to the "Transcribe" tab
3. Select the language for transcription
4. Click "Transcribe Audio Chunks"

The transcription will be saved to the `audio_text` directory, with both individual chunk files and a combined transcription file.

## Supported Languages

- Korean
- English
- Chinese
- Vietnamese
- Spanish

## Customization

You can customize the initial prompts for each language by editing the `whisper_prompt.json` file.

## Troubleshooting

If you encounter issues with the transcription:

1. Make sure whisper.cpp is properly compiled
2. Verify that the model file exists at the specified path
3. Check the logs for any error messages

## Dependencies

- whisper.cpp
- pydub
- gradio 