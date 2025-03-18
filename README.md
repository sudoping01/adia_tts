# Adia TTS - Wolof Text-to-Speech

A Python package for Text-to-Speech synthesis in the Wolof language powered by CONCREE's Adia_TTS model.

## Features

- High-quality Wolof language speech synthesis
- Intelligent text segmentation for natural-sounding long-form content
- Adaptive audio concatenation with smooth crossfading between segments
- Support for both single utterances and long-form text
- Simple, flexible API for integration into any Python application

## Installation

```bash
# Install from PyPI (when available)
pip install adia-tts

# Or install from GitHub
pip install git+https://github.com/CONCREE/adia-tts.git

# For development
git clone https://github.com/CONCREE/adia-tts.git
cd adia-tts
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

Fine-tune the speech generation with various parameters:

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
    text="Custom voice configuration example",
    description="A gentle male voice with clear articulation",
    config=config
)
```

## Advanced Usage

### GPU Acceleration

The package automatically uses CUDA if available:

```python
# Check device being used
tts = AdiaTTS()
device_info = tts.get_device_info()
print(f"Using: {device_info['device']}")
print(f"GPU: {device_info['gpu_name']}" if device_info['gpu_available'] else "CPU only")
```

### HuggingFace Authentication

For models requiring authentication:

```python
# Using environment variable
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token'
tts = AdiaTTS()

# Or pass directly
tts = AdiaTTS(hf_token="your_huggingface_token")
```

## Requirements

- Python 3.8+
- PyTorch
- transformers
- parler-tts
- soundfile
- numpy

## License

[MIT License](LICENSE)