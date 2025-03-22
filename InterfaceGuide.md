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
