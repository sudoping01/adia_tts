from setuptools import setup, find_packages

setup(
    name="adia_tts",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "numpy>=1.19.0",
        "soundfile>=0.10.0",
        "transformers>=4.35.0",
        "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
        "accelerate>=0.20.0"
    ],
    author="sudoping01",
    author_email="sudoping01@gmail.com",  
    description="adia tts inference package, that allow to avoid the the model limitation with easy inference",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md", "r", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/sudoping01/adia-tts",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Wolof",
    ],
    python_requires=">=3.8",
)

