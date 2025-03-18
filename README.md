# Adia TTS - Wolof Text-to-Speech
# Adia TTS

A Python package for Wolof text-to-speech synthesis.

## Overview
Adia TTS is the most accurate open-source text-to-speech model for the Wolof language, at the moment of writing this readme. This package provides easy inference capabilities while addressing the model's input size limitations. The core functionality includes a segmentation strategy that allows the model to handle long sentences, making it suitable for conversational agents and extended speech synthesis applications.

## Installation

```bash

pip install git+https://github.com/sudoping01/adia_tts.git 

# For development
git clone git+https://github.com/sudoping01/adia_tts.git
cd adia_tts
pip install -e .
```

## Quick Start

### Basic Text-to-Speech

```python
from adia_tts import AdiaTTS

# Initialize the model
tts = AdiaTTS()

# Synthesize speech
output_path, audio_array = tts.synthesize(
    text="Salaam aleekum, nanga def?",
    description="A warm and natural voice, with a conversational flow"
)

print(f"Audio saved to: {output_path}")
```

### Processing Long Text

The package automatically handles long text by breaking it at natural pauses:

```python
from adia_tts import AdiaTTS

# Initialize the model
tts = AdiaTTS()

# Process long text with automatic segmentation
combined_path, segment_paths = tts.batch_synthesize(
    text="Your long Wolof text here...",
    description="A warm and natural voice, with a conversational flow"
)

print(f"Combined audio saved to: {combined_path}")

# Optional: Clean up individual segment files
tts.cleanup_temp_files(segment_paths)
```

### Reading from a File

Process entire text files with a single function call:

```python
from adia_tts import AdiaTTS

# Initialize the model with custom output directory
tts = AdiaTTS(output_dir="./my_audio_files")

# Convert text file to speech
combined_path, segment_paths = tts.synthesize_from_file(
    file_path="your_text_file.txt",
    description="A warm and natural voice, with a conversational flow"
)

print(f"Audio saved to: {combined_path}")
```

## Customizing Voice Generation

the speech generation parameters:

```python
# Voice generation configuration
config = {
    "temperature": 0.01,       # Lower for more consistent output
    "max_new_tokens": 1000,    # Maximum length of generated audio
    "do_sample": True,         # Enable sampling for more natural speech
    "top_k": 50,               # Sampling from top k tokens
    "repetition_penalty": 1.2  # Reduces repetitive patterns
}

# Apply custom configuration
output_path, _ = tts.synthesize(
    text="your text here",
    description="A gentle male voice with clear articulation",
    config=config
)
```



[MIT License](LICENSE)