import os
import sys
import argparse

# 添加项目路径
sys.path.append('/data4/yanghan/Generation/MCA-Ctrl_git')

# CUDA_VISIBLE_DEVICES=3 python /data4/yanghan/Generation/MCA-Ctrl_demo/demo1.0.py

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

# 默认参数
DEFAULT_SELF_REPLACE_STEPS = 0.1
DEFAULT_SAQI_ST = 0
DEFAULT_START_LAYER = 0
DEFAULT_END_LAYER = 16
DEFAULT_START_STEP = 0
DEFAULT_END_STEP = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 4056

# 模型路径
SD_MODEL_PATH = '/data4/yanghan/Generation/HuggingFace/stabilityai/SG161222/Realistic_Vision_V4.0_noVAE'
GROUNDING_DINO_CONFIG = '/data4/yanghan/Generation/HuggingFace/Shilongliu/GroundingDINO/GroundingDINO_SwinB.cfg.py'
GROUNDING_DINO_CHECKPOINT = '/data4/yanghan/Generation/HuggingFace/Shilongliu/GroundingDINO/groundingdino_swinb_cogcoor.pth'
SAM_CHECKPOINT = '/data4/yanghan/Generation/MCA-Ctrl/MCA-Ctrl_git/sam_vit_h_4b8939.pth'

sd_model = None
tokenizer = None
groundingdino_model = None
sam = None
sam_predictor = None

# 基于generation.ipynb
def load_models():
    global sd_model, tokenizer, groundingdino_model, sam, sam_predictor
    
    if sd_model is not None:
        return
    
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                             clip_sample=False, set_alpha_to_one=False)
    sd_model = MasaCtrlPipeline_generation.from_pretrained(SD_MODEL_PATH, scheduler=scheduler).to(DEVICE)
    tokenizer = sd_model.tokenizer
    
    args = SLConfig.fromfile(GROUNDING_DINO_CONFIG)
    groundingdino_model = build_model(args)
    checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT, map_location='cpu')
    groundingdino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    groundingdino_model.eval()
    groundingdino_model.to(DEVICE)
    
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
    # 不使用with语句
    progress(0, desc="准备模型中...")
    load_models()
    
    if ref_image is None:
        return None, "请上传参考图像"
    
    progress(0.1, desc="设置参数中...")
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    ref_image_pil = Image.fromarray(ref_image) if isinstance(ref_image, np.ndarray) else ref_image
    ref_image_pil = ref_image_pil.convert('RGB')
    
    progress(0.2, desc="反演参考图像中...")
    ref_image_tensor = load_image_k(ref_image_pil, DEVICE)
    ref_prompt = ""
    
    ref_start_code, ref_latents_list = sd_model.invert(
        ref_image_tensor, ref_prompt, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, return_intermediates=True
    )
    
    progress(0.5, desc="检测目标对象中...")
    trans_ref = load_dino_image(ref_image_pil).to(DEVICE)
    
    ref_boxes, ref_logits, ref_phrases = predict(
        model=groundingdino_model, image=trans_ref, caption=subject_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold, device=DEVICE
    )
    
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
    
    progress(0.7, desc="生成新图像中...")
    prompts = [prompt, prompt]
    
    editor = McaControlGeneration(
        [2], DEFAULT_SELF_REPLACE_STEPS, DEFAULT_SAQI_ST, DEFAULT_START_STEP, 
        DEFAULT_END_STEP, DEFAULT_START_LAYER, DEFAULT_END_LAYER, mask_r=ref_mask.float()
    )
    
    regiter_attention_editor_diffusers(sd_model, editor)
    
    image_masactrl = sd_model(
        prompts, latents=None, num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, ref_intermediate_latents=ref_latents_list,
        neg_prompt="", mask_r=ref_mask.float()
    )
    
    output_pil = transforms.ToPILImage()(image_masactrl[-1:].squeeze())
    
    progress(1.0, desc="完成!")
    return output_pil, f"使用种子: {seed}"

def process_with_model_loading(ref_image, prompt, subject_prompt, num_inference_steps, guidance_scale, seed, box_threshold, text_threshold, progress=gr.Progress()):
    try:
        print("开始加载模型...")
        load_models()
        print("模型加载成功")
        
        print(f"开始生成，参数: 提示词='{prompt}', 种子={seed}")
        result = process_generation(ref_image, prompt, subject_prompt, num_inference_steps, guidance_scale, seed, box_threshold, text_threshold, progress)
        print("生成完成")
        
        return result[0], result[1], "模型已加载，点击生成图像继续使用"
    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        print(f"处理时出错: {e}\n{error_info}")
        return None, f"处理失败: {str(e)}", f"错误: {str(e)}"
        
# Gradio界面设计
def main():
    with gr.Blocks(title="MCA-Ctrl", css="footer {visibility: hidden}") as demo:
        gr.Markdown("# MCA-Ctrl 主体生成器")
        gr.Markdown("上传参考图像，输入提示词，并将原始主体放置到新场景中")
        
        # 创建一个状态指示器
        model_status = gr.Textbox(label="模型状态", value="模型未加载，生成时将自动加载", interactive=False)
        
        # 使用更对称的布局
        with gr.Row(equal_height=True):
            # 左侧：输入区域
            with gr.Column(scale=1):
                # 参考图像区域
                ref_image_input = gr.Image(
                    label="参考图像", 
                    type="pil", 
                    height=400,
                    value="/data4/yanghan/Generation/MCA-Ctrl_git/data/image1.png"
                )
                
                # 示例图像
                gr.Examples(
                    examples=[
                        "/data4/yanghan/Generation/MCA-Ctrl_git/data/image1.png",
                        "/data4/yanghan/Generation/MCA-Ctrl_git/data/image3.jpeg", 
                        "/data4/yanghan/Generation/MCA-Ctrl_git/data/image4.png",
                        "/data4/yanghan/Generation/MCA-Ctrl_git/data/image5.png"
                    ],
                    inputs=ref_image_input,
                    label="示例图像"
                )
                
                # 提示词指南
                gr.Markdown("### 提示词指南（对于动漫角色，推荐使用'anime'作为主体检测提示词）")
                
                # 输入文本字段
                subject_prompt = gr.Textbox(
                    label="主体检测提示词", 
                    value="anime", 
                    lines=1,
                    placeholder="例如：anime, person, cat, dog"
                )
                
                prompt = gr.Textbox(
                    label="生成提示词", 
                    value="一个动漫人物在城市河畔，樱花盛开的场景", 
                    lines=2,
                    placeholder="描述期望的场景，例如：在森林中，在海滩上"
                )
                
                # 提示词示例 - 移到输入字段下方
                gr.Examples(
                    examples=[
                        ["anime", "一个动漫人物在阳光照射的美丽森林中"],
                        ["anime", "一个动漫人物在日落时分的海滩上，有海浪"],
                        ["cat", "一只狗在开满鲜花的绿色草地上"]
                    ],
                    inputs=[subject_prompt, prompt],
                    label="提示词示例"
                )
                
                # 生成按钮
                generate_btn = gr.Button("生成图像", variant="primary", size="lg")
            
            # 输出区域
            with gr.Column(scale=1):
                # 输出图像
                output_image = gr.Image(label="生成的图像", height=400)
                output_info = gr.Textbox(label="生成信息", lines=1)
                
                # 使用技巧 - 移到图像下方
                gr.Markdown("### 使用技巧")
                gr.Markdown("""
                - 如果未检测到主体，请尝试调整检测阈值
                - 增加推理步数可获得更高质量的图像
                - 使用不同的种子获得不同的结果
                - 
                -

                """)
                
                # 生成设置
                gr.Markdown("### 生成设置")
                seed = gr.Slider(label="种子 (-1 表示随机)", minimum=-1, maximum=2147483647, step=1, value=DEFAULT_SEED)
                with gr.Row():
                    num_inference_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, value=DEFAULT_NUM_INFERENCE_STEPS, step=1)
                    guidance_scale = gr.Slider(label="引导尺度", minimum=1.0, maximum=20.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1)
                
                # 参数设置
                gr.Markdown("### 检测设置")
                with gr.Row():
                    box_threshold = gr.Slider(label="框检测阈值", minimum=0.1, maximum=0.9, value=0.3, step=0.05)
                    text_threshold = gr.Slider(label="文本检测阈值", minimum=0.1, maximum=0.9, value=0.25, step=0.05)


        # 修改process_generation函数以包含模型加载过程
        generate_btn.click(
            fn=process_with_model_loading,
            inputs=[
                ref_image_input,      # 参考图像
                prompt,               # 生成提示词
                subject_prompt,       # 主体检测提示词
                num_inference_steps,  # 推理步数
                guidance_scale,       # 引导尺度
                seed,                 # 种子
                box_threshold,        # 框阈值
                text_threshold        # 文本阈值
            ],
            outputs=[output_image, output_info, model_status]
        )
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    

    print("启动Gradio界面...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=args.port, 
        share=args.share,
        show_error=True   # 显示错误
    )
    print("Gradio界面已关闭")

if __name__ == "__main__":
    main()