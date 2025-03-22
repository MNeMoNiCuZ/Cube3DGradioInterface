# What is Cube3D Gradio Interface?
Cube3D Gradio Interface is a simple yet functional interface for generating one or more files with [Cube3D by Roblox](https://github.com/Roblox/cube).

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