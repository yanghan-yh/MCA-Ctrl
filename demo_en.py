import os
import sys
import argparse
import traceback

# Add the project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="MCA-Ctrl Demo")
    parser.add_argument('--path', type=str, default='.',
                       help='Base project path')
    parser.add_argument('--model-path', type=str, default='.',
                       help='Path to model directory')
    parser.add_argument('--sd-model', type=str, 
                       default='SG161222/Realistic_Vision_V4.0_noVAE', 
                       help='Path to Stable Diffusion model or Hugging Face model ID')
    parser.add_argument('--use-local-sd', action='store_true',
                       help='Use local SD model path instead of Hugging Face ID')
    parser.add_argument('--dino-config', type=str,
                       default='Shilongliu/GroundingDINO/GroundingDINO_SwinB.cfg.py',
                       help='Path to GroundingDINO config file')
    parser.add_argument('--dino-checkpoint', type=str,
                       default='Shilongliu/GroundingDINO/groundingdino_swinb_cogcoor.pth',
                       help='Path to GroundingDINO checkpoint file')
    parser.add_argument('--sam-checkpoint', type=str,
                       default='sam_vit_h_4b8939.pth',
                       help='Path to SAM checkpoint file')
    parser.add_argument('--data-dir', type=str,
                       default='/data4/yanghan/Generation/MCA-Ctrl_git/data',
                       help='Directory containing example images')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port for Gradio interface')
    parser.add_argument('--share', action='store_true',
                       help='Share the Gradio interface')
    return parser.parse_args()

# Parse arguments at the beginning
args = parse_args()

# Build paths
project_path = args.path
model_path = args.model_path

# Modify model path handling logic
if hasattr(args, 'use_local_sd') and args.use_local_sd:
    SD_MODEL_PATH = os.path.join(model_path, args.sd_model)
else:
    # Use model ID directly without appending path
    SD_MODEL_PATH = args.sd_model

# Other paths remain unchanged
GROUNDING_DINO_CONFIG = os.path.join(model_path, args.dino_config)
GROUNDING_DINO_CHECKPOINT = os.path.join(model_path, args.dino_checkpoint)
SAM_CHECKPOINT = os.path.join(project_path, args.sam_checkpoint)

# Example images directory - use absolute path
DATA_DIR = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(project_path, args.data_dir)

import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_lightning import seed_everything
import random

from diffusers import DDIMScheduler
from masactrl.diffuser_utils_subject_generation import MasaCtrlPipeline_generation
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl_p2p import McaControlGeneration
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import predict
from segment_anything import build_sam, SamPredictor


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Default parameters
DEFAULT_SELF_REPLACE_STEPS = 0.1
DEFAULT_SAQI_ST = 0
DEFAULT_START_LAYER = 0
DEFAULT_END_LAYER = 16
DEFAULT_START_STEP = 0
DEFAULT_END_STEP = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 4056
DEFAULT_NEGATIVE_PROMPT = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

sd_model = None
tokenizer = None
groundingdino_model = None
sam = None
sam_predictor = None

# Based on generation.ipynb
def load_models():
    global sd_model, tokenizer, groundingdino_model, sam, sam_predictor
    
    if sd_model is not None:
        return
    
    print(f"Loading Stable Diffusion model: {SD_MODEL_PATH}")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                             clip_sample=False, set_alpha_to_one=False)
    sd_model = MasaCtrlPipeline_generation.from_pretrained(SD_MODEL_PATH, scheduler=scheduler).to(DEVICE)
    tokenizer = sd_model.tokenizer
    
    print(f"Loading GroundingDINO config: {GROUNDING_DINO_CONFIG}")
    print(f"Loading GroundingDINO checkpoint: {GROUNDING_DINO_CHECKPOINT}")
    args_config = SLConfig.fromfile(GROUNDING_DINO_CONFIG)
    groundingdino_model = build_model(args_config)
    checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT, map_location='cpu')
    groundingdino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    groundingdino_model.eval()
    groundingdino_model.to(DEVICE)
    
    print(f"Loading SAM model: {SAM_CHECKPOINT}")
    sam = build_sam(checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

def load_image_k(image_pil, device):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    image = transform(image_pil).unsqueeze(0)
    image = (image * 2.0 - 1.0).to(device)
    return image

def load_dino_image(image_pil):
    transform = T.Compose([
        T.RandomResize([400], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    return image_transformed

def process_generation(ref_image, prompt, subject_prompt, num_inference_steps, guidance_scale, seed, box_threshold, text_threshold, progress=gr.Progress()):
    # Don't use with statement
    progress(0, desc="Preparing models...")
    load_models()
    
    if ref_image is None:
        return None, "Please upload a reference image"
    
    progress(0.1, desc="Setting parameters...")
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    ref_image_pil = Image.fromarray(ref_image) if isinstance(ref_image, np.ndarray) else ref_image
    ref_image_pil = ref_image_pil.convert('RGB')
    
    progress(0.2, desc="Inverting reference image...")
    ref_image_tensor = load_image_k(ref_image_pil, DEVICE)
    ref_prompt = ""
    
    ref_start_code, ref_latents_list = sd_model.invert(
        ref_image_tensor, ref_prompt, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, return_intermediates=True
    )
    
    progress(0.5, desc="Detecting target object...")
    trans_ref = load_dino_image(ref_image_pil).to(DEVICE)
    
    ref_boxes, ref_logits, ref_phrases = predict(
        model=groundingdino_model, image=trans_ref, caption=subject_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold, device=DEVICE
    )
    
    if ref_boxes is None or len(ref_boxes) == 0:
        return None, f"Subject '{subject_prompt}' not detected. Try adjusting thresholds or changing prompt."
    
    if ref_boxes.device != DEVICE:
        ref_boxes = ref_boxes.to(DEVICE)
    
    ref_image_np = np.array(ref_image_pil.resize((512, 512)))
    sam_predictor.set_image(np.uint8(ref_image_np))
    
    H, W, _ = ref_image_np.shape
    dim_tensor = torch.tensor([W, H, W, H], device=DEVICE)
    ref_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(ref_boxes) * dim_tensor
    
    ref_transformed_boxes = sam_predictor.transform.apply_boxes_torch(ref_boxes_xyxy, ref_image_np.shape[:2]).to(DEVICE)
    
    ref_masks, _, _ = sam_predictor.predict_torch(
        point_coords=None, point_labels=None, boxes=ref_transformed_boxes, multimask_output=False
    )
    
    ref_mask = torch.where(ref_masks[0][0], 1., 0.)
    
    progress(0.7, desc="Generating new image...")
    prompts = [prompt, prompt]
    
    editor = McaControlGeneration(
        [2], DEFAULT_SELF_REPLACE_STEPS, DEFAULT_SAQI_ST, DEFAULT_START_STEP, 
        DEFAULT_END_STEP, DEFAULT_START_LAYER, DEFAULT_END_LAYER, mask_r=ref_mask.float()
    )
    
    regiter_attention_editor_diffusers(sd_model, editor)
    
    image_masactrl = sd_model(
        prompts, latents=None, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, ref_intermediate_latents=ref_latents_list,
        neg_prompt=DEFAULT_NEGATIVE_PROMPT, mask_r=ref_mask.float()
    )
    
    output_pil = transforms.ToPILImage()(image_masactrl[-1:].squeeze())
    
    progress(1.0, desc="Done!")
    return output_pil, f"Using seed: {seed}"

def process_with_model_loading(ref_image, prompt, subject_prompt, num_inference_steps, guidance_scale, seed, box_threshold, text_threshold, progress=gr.Progress()):
    try:
        print("Starting to load models...")
        load_models()
        print("Models loaded successfully")
        
        print(f"Starting generation, parameters: prompt='{prompt}', seed={seed}")
        result = process_generation(ref_image, prompt, subject_prompt, num_inference_steps, guidance_scale, seed, box_threshold, text_threshold, progress)
        print("Generation complete")
        
        return result[0], result[1], "Models loaded, click Generate Image to continue"
    except Exception as e:
        error_info = traceback.format_exc()
        print(f"Error during processing: {e}\n{error_info}")
        return None, f"Processing failed: {str(e)}", f"Error: {str(e)}"
        
# Helper function to find images in a directory
def find_images_in_directory(directory):
    """Find all images in a directory with common image extensions"""
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    images = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return images
        
    for file in os.listdir(directory):
        extension = os.path.splitext(file.lower())[1]
        if extension in valid_extensions:
            image_path = os.path.join(directory, file)
            if os.path.isfile(image_path):
                images.append(image_path)
                print(f"Found example image: {image_path}")
    
    return images

# Gradio interface design
def main():
    # Find example image paths
    example_paths = [
        os.path.join(DATA_DIR, "image1.png"),
        os.path.join(DATA_DIR, "image3.jpeg"), 
        os.path.join(DATA_DIR, "image4.png"),
        os.path.join(DATA_DIR, "image5.png")
    ]
    
    # Find all images in data directory
    all_images = find_images_in_directory(DATA_DIR)
    for img in all_images:
        if img not in example_paths:
            example_paths.append(img)
    
    # Validate paths
    valid_examples = []
    for path in example_paths:
        if os.path.exists(path):
            valid_examples.append(path)
            print(f"Valid example image: {path}")
        else:
            print(f"Invalid example image path: {path}")
    
    default_image = valid_examples[0] if valid_examples else None
    
    with gr.Blocks(title="MCA-Ctrl", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# MCA-Ctrl Subject Generator")
        gr.Markdown("Upload a reference image, input prompts, and place the original subject into a new scene")
        
        # Create a status indicator
        model_status = gr.Textbox(label="Model Status", value="Models not loaded, will load automatically when generating", interactive=False)
        
        # Use a more symmetric layout
        with gr.Row(equal_height=True):
            # Left: Input area
            with gr.Column(scale=1):
                # Reference image area
                ref_image_input = gr.Image(
                    label="Reference Image", 
                    type="pil", 
                    height=400,
                    value=default_image
                )
                
                # Example images
                if valid_examples:
                    gr.Examples(
                        examples=valid_examples,
                        inputs=ref_image_input,
                        label="Example Images"
                    )
                else:
                    gr.Markdown("### No example images found, please upload your own")
                
                # Prompt guide
                gr.Markdown("### Prompt Guide (For anime characters, recommend using 'anime' as subject detection prompt)")
                
                # Input text fields
                subject_prompt = gr.Textbox(
                    label="Subject Detection Prompt", 
                    value="anime", 
                    lines=1,
                    placeholder="e.g., anime, person, cat, dog"
                )
                
                prompt = gr.Textbox(
                    label="Generation Prompt", 
                    value="an anime character at riverside of the city where cherry blossoms are in full bloom", 
                    lines=2,
                    placeholder="Describe the desired scene, e.g., in a forest, on the beach"
                )
                
                # Prompt examples - moved below input fields
                gr.Examples(
                    examples=[
                        ["anime", "an anime character in a beautiful forest with sunlight"],
                        ["anime", "an anime character on a beach at sunset with waves"],
                        ["cat", "a dog in a green meadow with flowers"]
                    ],
                    inputs=[subject_prompt, prompt],
                    label="Prompt Examples"
                )
                
                # Generate button
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
            
            # Output area
            with gr.Column(scale=1):
                # Output image
                output_image = gr.Image(label="Generated Image", height=400)
                output_info = gr.Textbox(label="Generation Info", lines=1)
                
                # Usage tips - moved below image
                gr.Markdown("### Usage Tips")
                gr.Markdown("""
                - If subject is not detected, try adjusting detection thresholds
                - Increase inference steps for higher quality images
                - Use different seeds for different results
                - You can change model paths using command line arguments
                - Use --use-local-sd parameter to use local model files
                """)
                
                # Generation settings
                gr.Markdown("### Generation Settings")
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=2147483647, step=1, value=DEFAULT_SEED)
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=DEFAULT_NUM_INFERENCE_STEPS, step=1)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1)
                
                # Parameter settings
                gr.Markdown("### Detection Settings")
                with gr.Row():
                    box_threshold = gr.Slider(label="Box Detection Threshold", minimum=0.1, maximum=0.9, value=0.3, step=0.05)
                    text_threshold = gr.Slider(label="Text Detection Threshold", minimum=0.1, maximum=0.9, value=0.25, step=0.05)

        # Modify process_generation function to include model loading process
        generate_btn.click(
            fn=process_with_model_loading,
            inputs=[
                ref_image_input,      # Reference image
                prompt,               # Generation prompt
                subject_prompt,       # Subject detection prompt
                num_inference_steps,  # Inference steps
                guidance_scale,       # Guidance scale
                seed,                 # Seed
                box_threshold,        # Box threshold
                text_threshold        # Text threshold
            ],
            outputs=[output_image, output_info, model_status]
        )

    print(f"Starting Gradio interface on port {args.port}...")
    print(f"Using model paths:")
    print(f"- SD Model: {SD_MODEL_PATH}")
    print(f"- DINO Config: {GROUNDING_DINO_CONFIG}")
    print(f"- DINO Checkpoint: {GROUNDING_DINO_CHECKPOINT}")
    print(f"- SAM Checkpoint: {SAM_CHECKPOINT}")
    print(f"- Data Directory: {DATA_DIR}")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=args.port, 
        share=args.share,
        show_error=True   # Show errors
    )
    print("Gradio interface closed")

if __name__ == "__main__":
    main()