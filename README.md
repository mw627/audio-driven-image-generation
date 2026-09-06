Audio-Driven Image Generation System

An end-to-end multimodal generation system that transforms environmental audio into corresponding visual scenes through audio understanding, LLM-based prompt adaptation, and diffusion-based image synthesis.

Overview

This project implements an Audio-to-Image generation pipeline that converts environmental sounds into visual scenes.

Given an input audio file, the system performs:

Audio Input → Audio Understanding → Scene Summarization → Visual Prompt Generation → Image Synthesis

The system integrates Qwen3-Omni, an LLM-based two-stage Prompt Adapter, and Stable Diffusion v1.5 into a unified Gradio web application.

⸻

Pipeline

┌──────────────────┐
│   Audio Input    │
│ Local File / URL │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Audio Validation │
│ & Preprocessing  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ Qwen3-Omni Audio         │
│ Captioning               │
│                          │
│ Audio → Audio Description│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Two-Stage Prompt Adapter │
│                          │
│ Stage 1: Semantic        │
│ Compression              │
│                          │
│ Stage 2: Visual Mapping  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Stable Diffusion v1.5    │
│                          │
│ Text Prompt → Image      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────┐
│ Generated Image  │
└──────────────────┘

⸻

Key Features

* Audio-to-Image Generation
    * Converts environmental sounds into corresponding visual scenes.
* Multi-Modal Audio Understanding
    * Uses Qwen3-Omni to extract semantic information from audio.
* Two-Stage Prompt Adapter
    * Converts audio descriptions into concise scene summaries.
    * Maps scene semantics into visually grounded Stable Diffusion prompts.
* Stable Diffusion Image Generation
    * Uses Stable Diffusion v1.5 for image synthesis.
    * Supports configurable inference steps, guidance scale, resolution, and random seed.
* Flexible Audio Input
    * Supports local audio files.
    * Supports audio input through HTTP/HTTPS URLs.
* Interactive Web Interface
    * Built with Gradio.
    * Integrates audio preprocessing, prompt generation, and image generation into a unified interface.

⸻

System Architecture

1. Audio Input & Preprocessing

The system accepts audio from either:

* Local audio files
* HTTP/HTTPS audio URLs

The preprocessing module performs:

* File format validation
* File size validation
* Audio duration checking
* URL downloading
* Long-audio trimming

Supported formats include:

.mp3
.wav
.aac
.flac
.ogg
.amr
.3gp
.3gpp
.m4a

⸻

2. Audio Captioning

The validated audio is processed by Qwen3-Omni to generate a semantic description of the auditory scene.

Conceptually:

Environmental Audio
        ↓
Qwen3-Omni
        ↓
Audio Semantic Description

For example:

Input:
Birds singing in a forest
Output:
A natural outdoor scene with birds singing among trees.

⸻

3. Two-Stage Prompt Adapter

A two-stage LLM-based Prompt Adapter is designed to bridge the gap between audio semantics and visual generation prompts.

Stage 1 — Semantic Compression

The audio description is transformed into a concise scene summary.

Audio Description
        ↓
Semantic Compression
        ↓
Concise Scene Summary

The purpose is to remove unnecessary audio-specific information while preserving the key environmental semantics.

Stage 2 — Visual Mapping

The scene summary is transformed into a Stable Diffusion-oriented visual prompt.

Scene Summary
      ↓
Visual Mapping
      ↓
Visual Generation Prompt

The generated prompt focuses on:

* Main subject
* Environment
* Atmosphere
* Lighting
* Visual details

This design helps reduce the semantic gap between auditory descriptions and visual representations.

⸻

Image Generation

The project uses:

Stable Diffusion v1.5

with the Hugging Face Diffusers library.

Default generation parameters:

Inference Steps: 30
Guidance Scale: 7.5
Resolution: 512 × 512

The system also supports:

* Custom image width
* Custom image height
* Custom inference steps
* Custom guidance scale
* Random seed
* Reproducible image generation

⸻

Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/audio-driven-image-generation.git
cd audio-driven-image-generation

2. Create a Python environment

Python 3.10 is recommended.

conda create -n audio2img python=3.10
conda activate audio2img

3. Install dependencies

pip install -r requirements.txt

⸻

API Configuration

The project uses DashScope-compatible APIs for multimodal audio understanding and LLM-based prompt adaptation.

Create a .env file:

DASHSCOPE_API_KEY=your_api_key_here

Alternatively, configure the environment variable directly:

export DASHSCOPE_API_KEY="your_api_key_here"

Do not commit your API key to GitHub.

⸻

Model Requirements

The system uses:

Audio Captioning

Qwen3-Omni

Prompt Adaptation

Qwen3-Next

Image Generation

Stable Diffusion v1.5

The Stable Diffusion model is downloaded automatically through Diffusers when required.

Large model weights and local model caches should not be committed to this repository.

⸻

Running the Application

Start the Gradio application:

python app.py

The interface will be available at:

http://localhost:7860

For a remote GPU server, you can access the Gradio interface through the corresponding port forwarding or server URL.

⸻

Web Interface

The web interface provides:

Input

* Local audio upload
* Audio URL input

Generation Parameters

* Inference steps
* Guidance scale
* Image width
* Image height
* Random seed

Output

* Generated Stable Diffusion prompt
* Generated image

⸻

Example

Audio
  ↓
Rain falling in a forest
  ↓
Audio Understanding
  ↓
Scene Summary
  ↓
A quiet forest during rainfall
  ↓
Visual Prompt
  ↓
Stable Diffusion
  ↓
Generated Forest Image

Example results can be found in:

assets/examples/

⸻

Technologies

Component	Technology
Programming Language	Python
Audio Understanding	Qwen3-Omni
Prompt Adaptation	Qwen3-Next
Image Generation	Stable Diffusion v1.5
Diffusion Framework	Hugging Face Diffusers
Web Interface	Gradio
API	DashScope
Deep Learning	PyTorch

⸻

Limitations

The current system has several limitations:

1. Audio-to-image semantic alignment
    * The generated image depends on the quality of the audio understanding and prompt adaptation stages.
2. Ambiguous audio scenes
    * Complex or overlapping sounds may lead to incomplete or ambiguous scene descriptions.
3. Visual generation limitations
    * Stable Diffusion may generate visually plausible scenes that do not perfectly correspond to the original audio.
4. Inference cost
    * The complete pipeline requires both multimodal LLM inference and diffusion-based image generation.
5. Hardware requirements
    * Local image generation benefits from a CUDA-enabled GPU with sufficient VRAM.

⸻

Future Work

Potential improvements include:

* More accurate audio-visual semantic alignment
* Multi-event audio scene decomposition
* Temporal audio analysis
* Audio-conditioned diffusion models
* Improved prompt optimization
* Fine-tuning on paired audio-image datasets
* Quantized model inference for lower GPU memory usage
* More controllable image generation
* Objective evaluation of audio-image semantic consistency

⸻

License

This project is intended for academic and research purposes.

See LICENSE for details.
