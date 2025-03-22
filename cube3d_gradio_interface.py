import os
import torch
import trimesh
import gradio as gr
import logging
import datetime
import numpy as np
import sys
from io import StringIO
from contextlib import contextmanager
from typing import List, Union, Tuple, Generator, Dict, Any
from cube3d.inference.engine import Engine, EngineFast

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = stdout

# Suppress Warp initialization output
with suppress_stdout():
    logging.getLogger("warp").setLevel(logging.WARNING)

class Cube3DGenerator:
    def __init__(self):
        self.config_path = "cube3d/configs/open_model.yaml"
        self.gpt_ckpt_path = "model_weights/shape_gpt.safetensors"
        self.shape_ckpt_path = "model_weights/shape_tokenizer.safetensors"
        self.output_dir = "outputs"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = None
        self.last_generation = {"prompt": "", "seed": None}
        self.progress_callback = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_engine(self):
        """Initialize the Cube3D engine based on CUDA availability"""
        if self.engine is not None:
            return
            
        try:
            with suppress_stdout():
                if torch.cuda.is_available():
                    self.engine = EngineFast(
                        self.config_path,
                        self.gpt_ckpt_path,
                        self.shape_ckpt_path,
                        device=self.device
                    )
                    # Ensure CUDA is properly initialized
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                else:
                    self.engine = Engine(
                        self.config_path,
                        self.gpt_ckpt_path,
                        self.shape_ckpt_path,
                        device=self.device
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize engine: {str(e)}")
        
        # Warm up the engine with a dummy generation
        try:
            with suppress_stdout():
                _ = self.engine.t2s(["warmup"], use_kv_cache=True, resolution_base=8.0)
        except Exception:
            pass

    def sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filenames"""
        # Replace illegal filename characters
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            text = text.replace(char, '_')
        # Limit length and remove leading/trailing spaces
        return text.strip()[:50]

    def parse_filename_template(self, template: str, index: int, prompt: str, seed: int, total_count: int) -> str:
        """Parse filename template and replace variables"""
        # Get timestamp
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d - %H.%M.%S")
        
        # Prepare variables
        variables = {
            '[timestamp]': timestamp,
            '[variation]': f"{index + 1}" if total_count > 1 else '',
            '[prompt]': self.sanitize_filename(prompt),
            '[seed]': str(seed) if seed != -1 else 'random'
        }
        
        # Replace variables in template
        result = template
        for var, value in variables.items():
            if value:
                result = result.replace(var, value)
        
        # Remove any remaining brackets and clean up multiple underscores
        result = result.replace('[', '').replace(']', '')
        result = '_'.join(filter(None, result.split('_')))
        
        return result if result else 'generated'

    def get_output_path(self, batch_index: int, template: str = "[timestamp]", 
                       prompt: str = "", seed: int = -1, total_count: int = 1) -> str:
        """Generate output path with template-based naming"""
        # Create date folder
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        folder_path = os.path.join(self.output_dir, date_folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Parse template and generate filename
        filename = self.parse_filename_template(
            template, batch_index, prompt, seed, total_count
        )
        
        # Return base path without extension (we'll add .glb and .obj later)
        return os.path.join(folder_path, filename)

    def set_seed(self, seed: int = -1) -> int:
        """Set all random seeds and return the seed used"""
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Set seeds for all random number generators
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        return seed

    def set_progress_callback(self, callback):
        """Set a callback function for progress updates"""
        self.progress_callback = callback
    
    def generate_single(self, prompt: str, index: int = 0, seed: int = -1, 
                       template: str = "[timestamp]", total_count: int = 1, 
                       resolution_base: float = 8.0, append_seed: bool = True, 
                       save_prompt: bool = False, save_obj: bool = True, save_glb: bool = True) -> Tuple[str, str, trimesh.Trimesh, dict]:
        """Generate a single 3D model from a prompt"""
        # Set seed
        used_seed = self.set_seed(seed)
        
        # Append seed to prompt if requested
        generation_prompt = prompt
        if append_seed:
            generation_prompt = f"{prompt} [seed:{used_seed}]"
        
        # Generate the mesh
        mesh_v_f = self.engine.t2s([generation_prompt], use_kv_cache=True, resolution_base=resolution_base)
        vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
        
        # Get output path and save mesh
        base_path = self.get_output_path(
            index, template, prompt, used_seed, total_count
        )
        glb_path = f"{base_path}.glb"
        obj_path = f"{base_path}.obj"
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.process()
        mesh.fix_normals()
        if save_glb:
            mesh.export(glb_path, file_type="glb")
        if save_obj:
            mesh.export(obj_path, file_type="obj", include_normals=True)
        
        # Save prompt to text file if requested
        if save_prompt:
            text_path = f"{base_path}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        # Store generation info
        self.last_generation = {
            "prompt": prompt,
            "seed": used_seed
        }
        
        return glb_path, obj_path, mesh, self.last_generation

    def generate_batch(self, prompts: Union[str, List[str]], count: int = 1, 
                      seed: int = -1, template: str = "[timestamp]",
                      multi_prompt: bool = False, resolution_base: float = 8.0,
                      append_seed: bool = True, save_prompt: bool = False,
                      save_obj: bool = True, save_glb: bool = True) -> Generator[Tuple[str, str, trimesh.Trimesh, dict], None, None]:
        """Generate multiple 3D models from prompt(s)"""
        self.initialize_engine()
        
        # Convert input to list of prompts
        if isinstance(prompts, str):
            prompts = [line.strip() for line in prompts.split('\n') if line.strip()]
        else:
            prompts = [p.strip() for p in prompts if p.strip()]
        
        if not prompts:
            return
        
        # Calculate total count for filename numbering
        total_count = len(prompts) * count if multi_prompt else count
        current_index = 0
        
        if multi_prompt:
            # Generate count times for each prompt
            for prompt in prompts:
                for i in range(count):
                    current_seed = seed if seed != -1 else -1
                    glb_path, obj_path, mesh, gen_info = self.generate_single(
                        prompt, current_index, current_seed,
                        template, total_count, resolution_base,
                        append_seed, save_prompt, save_obj, save_glb
                    )
                    current_index += 1
                    yield glb_path, obj_path, mesh, gen_info
        else:
            # Generate count times for the first prompt
            prompt = prompts[0]
            for i in range(count):
                current_seed = seed if seed != -1 else -1
                glb_path, obj_path, mesh, gen_info = self.generate_single(
                    prompt, i, current_seed,
                    template, total_count, resolution_base,
                    append_seed, save_prompt, save_obj, save_glb
                )
                yield glb_path, obj_path, mesh, gen_info

def process_input(text: str, count: int, seed: int, template: str, 
                 multi_prompt: bool, resolution_base: float, scale_percent: int,
                 rotate_x: float, rotate_y: float, rotate_z: float,
                 append_seed: bool, save_prompt: bool, save_obj: bool, save_glb: bool,
                 model_list: List[Dict[str, str]], selected_model: str) -> Generator[List[Union[str, str, str, str, List[Tuple[str, str]], List[Dict[str, str]], str]], None, None]:
    """Process input text and generate 3D models"""
    # Calculate total generations
    num_prompts = len([line for line in text.split('\n') if line.strip()])
    total_gens = num_prompts * count if multi_prompt else count
    
    # Initialize outputs
    model_updates = [gr.update(visible=False) for _ in range(10)]
    info_updates = [""] * 10
    
    # Update status with proper pluralization
    status = "Starting model generation..." if total_gens == 1 else f"Starting generation of {total_gens} models..."
    updated_model_list = model_list.copy() if model_list else []
    model_options = [(f"{m['info'].split(' | ')[0]} - {m['info'].split(' | ')[2].split(': ')[1]} - {m['info'].split(' | ')[1].split(': ')[1]}" , m["glb_path"]) for m in updated_model_list]
    initial_model = selected_model if selected_model and os.path.exists(selected_model) else None
    yield [status, *model_updates, *info_updates, initial_model, gr.update(choices=model_options, value=initial_model), updated_model_list, initial_model]
    
    generator = Cube3DGenerator()
    
    # Generate models
    current_model = 0
    for glb_path, obj_path, mesh, gen_info in generator.generate_batch(
        text, count, seed, template, multi_prompt, 
        resolution_base, append_seed, save_prompt, save_obj, save_glb
    ):
        current_model += 1
        
        # Update status with proper pluralization
        status = f"Generating model {current_model} of {total_gens}..."
        
        # Apply transformations to the mesh directly
        if scale_percent != 100:
            scale_factor = scale_percent / 100.0
            mesh.vertices *= scale_factor
            
        # Apply rotations (in degrees)
        if any([rotate_x, rotate_y, rotate_z]):
            rotation_matrix = trimesh.transformations.euler_matrix(
                np.radians(rotate_x),
                np.radians(rotate_y),
                np.radians(rotate_z)
            )
            mesh.apply_transform(rotation_matrix)
            
        # Save processed mesh based on options
        if save_glb:
            mesh.export(glb_path, file_type="glb")
        if save_obj:
            mesh.export(obj_path, file_type="obj", include_normals=True)
        
        # Add to dynamic model list with timestamp, only if GLB is saved (for viewer)
        if save_glb:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"{timestamp} | Prompt: {gen_info['prompt']} | Seed: {gen_info['seed']}"
            updated_model_list.append({"glb_path": glb_path, "info": info_text})
            model_options = [(f"{m['info'].split(' | ')[0]} - {m['info'].split(' | ')[2].split(': ')[1]} - {m['info'].split(' | ')[1].split(': ')[1]}", m["glb_path"]) for m in updated_model_list]
            
            # Update all outputs, showing the latest model in Model3D
            yield [status, *model_updates, *info_updates, glb_path, gr.update(choices=model_options, value=glb_path), updated_model_list, glb_path]
        else:
            # If GLB isn't saved, don't update viewer or dropdown
            yield [status, *model_updates, *info_updates, None, gr.update(choices=model_options, value=None), updated_model_list, None]
    
    # Final status with proper pluralization
    status = "Model generation complete!" if total_gens == 1 else f"Generation complete! Created {total_gens} models"
    last_model = updated_model_list[-1]["glb_path"] if (save_glb and updated_model_list) else None
    model_options = [(f"{m['info'].split(' | ')[0]} - {m['info'].split(' | ')[2].split(': ')[1]} - {m['info'].split(' | ')[1].split(': ')[1]}", m["glb_path"]) for m in updated_model_list] if save_glb else []
    yield [status, *model_updates, *info_updates, last_model, gr.update(choices=model_options, value=last_model), updated_model_list, last_model]

def load_model(selected_path: str) -> str:
    """Check if the model file exists and return it or an error message"""
    if selected_path and os.path.exists(selected_path):
        return selected_path
    return "Failed to load model"

# Create Gradio interface
with gr.Blocks(title="Cube3D Generator Gradio Interface", theme=gr.themes.Default()) as demo:
    gr.Markdown("# Cube3D Generator Gradio Interface - [GitHub Repository](https://github.com/MNeMoNiCuZ/Cube3DGradioInterface)")
    
    with gr.Row():
        # Left column - Controls
        with gr.Column(scale=1, variant="panel"):
            # Prompt Settings
            gr.Markdown("### Prompt Settings")
            with gr.Group():
                prompt_input = gr.Textbox(
                    label="Prompt Text",
                    placeholder="Enter your prompt here. You can write multiple lines for a longer prompt.",
                    lines=5,
                    info="Enter your text prompt. Multiple lines will be treated as a single prompt unless 'Process each line separately' is enabled."
                )
                with gr.Row():
                    multi_prompt = gr.Checkbox(
                        label="Process each line separately",
                        value=False,
                        info="If enabled, each line will be treated as a separate prompt and generated individually."
                    )
                    append_seed = gr.Checkbox(
                        label="Add seed to prompt",
                        value=True,
                        info="Append the seed to the prompt for slight random variations."
                    )
            
            # Generation Settings
            gr.Markdown("### Generation Settings")
            with gr.Group():
                with gr.Row():
                    count = gr.Number(
                        label="Number of Variations",
                        value=1,
                        minimum=1,
                        maximum=100000,
                        step=1,
                        info="How many variations to generate from each prompt. Will be applied to each line if 'Process each line separately' is enabled."
                    )
                    seed = gr.Number(
                        label="Random Seed",
                        value=-1,
                        minimum=-1,
                        step=1,
                        info="Set a specific seed for reproducible results. Use -1 for random generation each time. Placeholder functionality."
                    )
                resolution_base = gr.Slider(
                    label="Resolution Base",
                    minimum=4.0,
                    maximum=9.0,
                    value=8.0,
                    step=0.5,
                    info="Controls mesh resolution. Lower values create coarser meshes but faster generation. Higher values give better quality but slower generation."
                )
            
            # Output Settings
            gr.Markdown("### Output Settings")
            with gr.Blocks(elem_classes="no-background"):
                # Filename template with variables description
                with gr.Row(elem_classes="no-background"):
                    with gr.Column(scale=2, elem_classes="no-background"):
                        template = gr.Textbox(
                            label="Filename Template",
                            placeholder="[timestamp]",
                            value="[timestamp]",
                            info="Combine text and variables to create custom filenames",
                            lines=1,
                            show_copy_button=False
                        )
                    with gr.Column(scale=1, elem_classes="no-background"):
                        gr.Textbox(
                            label="Available Variables",
                            value="[timestamp]: YYYY-MM-DD - HH.MM.SS\n[prompt]: Text used for generation\n[seed]: Generation seed\n[variation]: Variation index",
                            lines=4,
                            interactive=False,
                            show_copy_button=False,
                            show_label=True
                        )
                
                # Save options
                with gr.Row(elem_classes="no-background"):
                    save_obj = gr.Checkbox(
                        label="Save OBJ",
                        value=True,
                        info="Save the model as an OBJ file"
                    )
                    save_glb = gr.Checkbox(
                        label="Save GLB",
                        value=True,
                        info="Save the model as a GLB file (required for the model viewer)"
                    )
                
                # Model adjustments in rows
                with gr.Row(elem_classes="no-background"):
                    scale_percent = gr.Number(
                        label="Scale (%)",
                        value=2000,
                        minimum=1,
                        step=1,
                        info="Scale the generated model by this percentage"
                    )
                    save_prompt = gr.Checkbox(
                        label="Create caption from prompt",
                        value=False,
                        info="Save the prompt text in a caption file"
                    )
                
                with gr.Row(elem_classes="no-background"):
                    rotate_x = gr.Number(
                        label="X Rotation",
                        value=0,
                        minimum=-180,
                        maximum=180,
                        step=1,
                        info="Rotate around X-axis (degrees)"
                    )
                    rotate_y = gr.Number(
                        label="Y Rotation",
                        value=0,
                        minimum=-180,
                        maximum=180,
                        step=1,
                        info="Rotate around Y-axis (degrees)"
                    )
                    rotate_z = gr.Number(
                        label="Z Rotation",
                        value=0,
                        minimum=-180,
                        maximum=180,
                        step=1,
                        info="Rotate around Z-axis (degrees)"
                    )
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        # Right column - Model viewers and status
        with gr.Column(scale=1, variant="panel"):
            # Status console (at the top)
            status_output = gr.Textbox(
                label="Generation Status",
                interactive=False,
                lines=3
            )
            
            # Generated Models heading
            gr.Markdown("### Generated Models")
            
            # Single Model3D for viewing, with dropdown below
            model_display = gr.Model3D(label="Selected Model", height=400)
            model_dropdown = gr.Dropdown(label="Select Model", choices=[], value=None)
            model_list_state = gr.State(value=[])
            selected_model_state = gr.State(value=None)

    # Remove the scrollable container CSS
    demo.css = ""

    # Event handlers
    generate_btn.click(
        fn=process_input,
        inputs=[
            prompt_input, count, seed, template, 
            multi_prompt, resolution_base,
            scale_percent, rotate_x, rotate_y, rotate_z,
            append_seed, save_prompt, save_obj, save_glb,
            model_list_state, selected_model_state
        ],
        outputs=[
            status_output, 
            *([gr.Model3D(visible=False)] * 10), 
            *([gr.Markdown(visible=False)] * 10), 
            model_display, 
            model_dropdown, 
            model_list_state, 
            selected_model_state
        ]
    )

    # Handle dropdown selection
    model_dropdown.change(
        fn=load_model,
        inputs=model_dropdown,
        outputs=model_display
    )

if __name__ == "__main__":
    # IMPORTANT: Never enable share=True as it exposes the interface publicly!
    demo.launch(share=False)
