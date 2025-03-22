# Adia TTS - Wolof Text-to-Speech

A Python package for high-quality Wolof speech synthesis.

## Overview
[Adia TTS](https://huggingface.co/CONCREE/Adia_TTS) is the most accurate open-source text-to-speech model for the Wolof language, at the moment of writing this readme. This package provides easy inference capabilities while addressing the model's input size limitations. The core functionality includes a segmentation strategy that allows the model to handle long sentences, making it suitable for conversational agents and extended speech synthesis applications.

## Installation

```bash
# Standard installation
pip install git+https://github.com/sudoping01/adia_tts.git 

# For development
git clone git+https://github.com/sudoping01/adia_tts.git
cd adia_tts
pip install -e .
```

## Quick Start

### Basic Usage

```python
from adia_tts import AdiaTTS

# Initialize model
tts = AdiaTTS() # you can define your output_dir like tts = AdiaTTS(output_dir = "/home/sudoping01/audios")

# Generate speech
output_path, audio_array = tts.synthesize(
    text="Entreprenariat ci Senegal dafa am solo lool ci yokkuteg koom-koom, di gëna yokk liggéey ak indi gis-gis yu bees ci dëkk bi."

,
    description= "A clear and educational voice, with a flow adapted to learning"
)
print(f"Audio saved to: {output_path}")
```

### Long Text Processing

```python
# Automatic segmentation for long text
output_path, _ = tts.synthesize(
    text="Your long Wolof text here...",
    description= "A clear and educational voice, with a flow adapted to learning"
)
```

### File Processing

```python
# Process text files directly
tts = AdiaTTS(output_dir="./my_audio_files")
audio_path = tts.synthesize_from_file(
    file_path="your_text_file.txt"
)
```

## Customization

```python
# Voice configuration options
config = {
    "temperature": 0.01,       # Lower = more consistent
    "max_new_tokens": 1000,    # Maximum audio length
    "do_sample": True,         # Enable natural speech sampling
    "top_k": 50,               # Sampling parameter
    "repetition_penalty": 1.2  # Prevent repetition
}

# Apply custom settings
output_path, _ = tts.synthesize(
    text="Your text here",
    description="A gentle male voice",
    config=config
)
```

### Environment Information

```python
# Check runtime environment
device_info = tts.get_device_info()
print(f"Running on: {device_info['device']}")
print(f"GPU available: {device_info['gpu_available']}")
```


## License

[MIT License](LICENSE)
