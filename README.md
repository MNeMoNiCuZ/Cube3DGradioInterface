# What is Cube3D Gradio Interface?
Cube3D Gradio Interface is a simple yet functional interface for generating one or more files with [Cube3D by Roblox](https://github.com/Roblox/cube).

You can also try the model in their online demo interface: [https://huggingface.co/spaces/Roblox/cube3d-interactive](https://huggingface.co/spaces/Roblox/cube3d-interactive)

![image](https://github.com/user-attachments/assets/c79c2d31-0df2-4881-983a-463cb877457f)

# Installation Instructions
1. Git clone this repository 
  
`git clone [https://github.com/Roblox/cube](https://github.com/MNeMoNiCuZ/Cube3DGradioInterface)`

2. Create a virtual environment. You can use the `venv_create.bat` file in this repo to guide you through the process.

3. Follow the installation instructions of the original model:

Run the following commands inside the virtual environment:
`pip install -e .[meshlab]`
`huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights`

4. Install [pytorch](https://pytorch.org/) inside the virtual environment:

`pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`

5. Install the requirements in the virtual environment `pip install -r requirements.txt`

or manually install them: `pip install trimesh gradio numpy warp-lang`

# Running the interface
1. Launch the `cube_gradio_interface.py` from inside the environment. You can also run the `cube_gradio_interface.bat` to activate the venv and launch the interface.

# Interface
## Quick Start

1. Enter a text description in the "Prompt Text" box
2. Click "Generate"
3. Wait for generation to complete
4. View your model in the 3D viewer
5. Find generated files in the `outputs/[date]` folder

## Interface Features

### Prompt Settings
- **Prompt Text**: Describe the 3D model you want to generate
- **Process each line separately**: Generate a new / separate model for each line in the prompt
- **Add seed to prompt**: This will add your seed to the end of your prompt for randomness

### Generation Settings
- **Number of Variations**: How many versions to generate (default: 1)
- **Random Seed**: Set specific seed for variability or reproducable results (-1 for random), placeholder seed technique
- **Resolution Base**: Controls mesh quality (8.0 recommended, higher = better quality but slower)

### Output Settings
- **Filename Template**: Customize output filenames using variables:
  - [timestamp]: Date and time
  - [prompt]: Your input text
  - [seed]: Generation seed
  - [variation]: Variation number
- **Save Options**:
  - OBJ: Standard 3D format
  - GLB: Required for web viewer
- **Scale**: Adjust model size (default: 2000%)
- **Rotation**: Adjust X/Y/Z rotation in degrees
- **Create caption**: Save prompt text in separate file next to your model

## Model Viewer
- View generated models in real-time
- Select previous models from dropdown
- Models are saved in `outputs/[current_date]` folder


# Original Project Documentation below

# Cube: Generative AI System for 3D

<p align="center">
  <img src="./resources/teaser.png" width="800" style="margin: 5px;">
</p>

<div align="center">
  <a href=https://corp.roblox.com/newsroom/2025/03/introducing-roblox-cube target="_blank"><img src=https://img.shields.io/badge/Roblox-Blog-000000.svg?logo=Roblox height=22px></a>
  <a href=https://arxiv.org/abs/2503.15475 target="_blank"><img src=https://img.shields.io/badge/ArXiv-Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://huggingface.co/Roblox/cube3d-0.1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-d96902.svg height=22px></a>
  <a href=https://huggingface.co/spaces/Roblox/cube3d-interactive target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-blue.svg height=22px></a>
  <a href=https://colab.research.google.com/drive/1ZvTj49pjDCD_crX5WPZNTAoTTzL6-E5t target="_blank"><img src=https://img.shields.io/badge/Colab-Demo-blue.svg?logo=googlecolab height=22px></a>
</div>


Foundation models trained on vast amounts of data have demonstrated remarkable reasoning and
generation capabilities in the domains of text, images, audio and video. Our goal is to build
such a foundation model for 3D intelligence, a model that can support developers in producing all aspects
of a Roblox experience, from generating 3D objects and scenes to rigging characters for animation to
producing programmatic scripts describing object behaviors. As we start open-sourcing a family of models 
towards this vision, we hope to engage others in the research community to address these goals with us.

## Get Started with Cube 3D

<p align="center">
  <img src="./resources/greyscale_512.gif" width="600" style="margin: 5px;">
</p>

Cube 3D is our first step towards 3D intelligence, which involves a shape tokenizer and a text-to-shape generation model. We are unlocking the power of generating 3D assets and enhancing creativity for all artists. Our latest version of Cube 3D is now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly. This release includes model weights and starting code for using our text-to-shape model to create 3D assets.

### Try it out on 

- [Google Colab](https://colab.research.google.com/drive/1ZvTj49pjDCD_crX5WPZNTAoTTzL6-E5t)
- [Hugging Face Interactive Demo](https://huggingface.co/spaces/Roblox/cube3d-interactive)

### Install Requirements

Clone and install this repo in a virtual environment, via:

```bash
git clone https://github.com/Roblox/cube.git
cd cube
pip install -e .[meshlab]
```

> **CUDA**: If you are using a Windows machine, you may need to install the [CUDA](https://developer.nvidia.com/cuda-downloads) toolkit as well as `torch` with cuda support via `pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`

> **Note**: `[meshlab]` is an optional dependency and can be removed by simply running `pip install -e .` for better compatibility but mesh simplification will be disabled.

### Download Models from Huggingface 🤗

Download the model weights from [hugging face](https://huggingface.co/Roblox/cube3d-v0.1) or use the
`huggingface-cli`:

```bash
huggingface-cli download Roblox/cube3d-v0.1 --local-dir ./model_weights
```

### Inference

#### 1. Shape Generation

To generate 3D models using the downloaded models simply run:

```bash
python -m cube3d.generate \
            --gpt-ckpt-path model_weights/shape_gpt.safetensors \
            --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
            --fast-inference \
            --prompt "Broad-winged flying red dragon, elongated, folded legs."
```

> **Note**: `--fast-inference` is optional and may not be available for all GPU that have limited VRAM. This flag will also not work on MacOS. 

The output will be an `.obj` file saved in the specified `output` directory.

If you want to render a turntable gif of the mesh, you can use the `--render-gif` flag, which will render a turntable gif of the mesh
and save it as `turntable.gif` in the specified `output` directory. 

We provide several example output objects and their corresponding text prompts in the `examples` folder.

> **Note**: You must have Blender (version >= 4.3) installed and available in your system's PATH to render the turntable GIF. You can download it from [Blender's official website](https://www.blender.org/). Ensure that the Blender executable is accessible from the command line.

> **Note**: If shape decoding is slow, you can try to specify a lower resolution using the `--resolution-base` flag. A lower resolution will create a coarser and lower quality output mesh but faster decoding. Values between 4.0 and 9.0 are recommended.

#### 2. Shape Tokenization and De-tokenization

To tokenize a 3D shape into token indices and reconstruct it back, you can use the following command:

```bash
python -m cube3d.vq_vae_encode_decode \
            --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
            --mesh-path ./outputs/output.obj
```

This will process the `.obj` file located at `./outputs/output.obj` and prints the tokenized representation as well as exports the mesh reconstructed from the token indices.

### Hardware Requirements

We have tested our model on:
* Nvidia H100 GPU
* Nvidia A100 GPU
* Nvidia Geforce 3080
* Apple Silicon M2-4 Chips.

We recommend using a GPU with at least 24GB of VRAM available when using `--fast-inference` (or `EngineFast`) and 16GB otherwise. 

### Code Usage

We have designed a minimalist API that allows the use this repo as a Python library:

```python
import torch
import trimesh
from cube3d.inference.engine import Engine, EngineFast

# load ckpt
config_path = "cube3d/configs/open_model.yaml"
gpt_ckpt_path = "model_weights/shape_gpt.safetensors"
shape_ckpt_path = "model_weights/shape_tokenizer.safetensors"
engine_fast = EngineFast( # only supported on CUDA devices, replace with Engine otherwise
    config_path, 
    gpt_ckpt_path, 
    shape_ckpt_path, 
    device=torch.device("cuda"),
)

# inference
input_prompt = "A pair of noise-canceling headphones"
# NOTE: Reduce `resolution_base` for faster inference and lower VRAM usage
# The `top_p` parameter controls randomness between inferences:
#   Float < 1: Keep smallest set of tokens with cumulative probability ≥ top_p. Default None: deterministic generation.
mesh_v_f = engine_fast.t2s([input_prompt], use_kv_cache=True, resolution_base=8.0, top_p=0.9)

# save output
vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
_ = trimesh.Trimesh(vertices=vertices, faces=faces).export("output.obj")
```

## Coming Soon

### Controlling shape generation with bounding box conditioning
<div align="center">
  <img src="./resources/truck_black_text_512.gif" width="300" height="300" style="margin: 5px;">
  <img src="./resources/couch_black_text_512.gif" width="300" height="300" style="margin: 5px;">
</div>

### Scene Generation

https://github.com/user-attachments/assets/987c459a-5708-41a5-9b92-89068a70a239

https://github.com/user-attachments/assets/ab501a86-b0cb-4c73-827e-988b2120d4c0

## Citation
If you find this work helpful, please consider citing our technical report:

```bibtex
@article{roblox2025cube,
    title   = {Cube: A Roblox View of 3D Intelligence},
    author  = {Roblox, Foundation AI Team},
    journal = {arXiv preprint arXiv:2503.15475},
    year    = {2025}
}
```

## Acknowledgements

We would like to thank the contributors of [TRELLIS](https://github.com/microsoft/TRELLIS), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), [threestudio](https://github.com/threestudio-project/threestudio), [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [minGPT](https://github.com/karpathy/minGPT), [dinov2](https://github.com/facebookresearch/dinov2), [OptVQ](https://github.com/zbr17/OptVQ), [1d-tokenizer](https://github.com/bytedance/1d-tokenizer)
repositories, for their open source contributions.
