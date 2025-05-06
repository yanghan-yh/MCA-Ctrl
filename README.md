## MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization

Pytorch implementation of MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization

---

## Introduction

![intro](fig/intro.png)

The rapid advancement of diffusion models has increased the need for customized image generation. However, current customization methods face several limitations: 1) typically accept either image or text conditions alone; 2) customization in complex visual scenarios often leads to subject leakage or confusion; 3) image-conditioned outputs tend to suffer from inconsistent backgrounds; and 4) high computational costs.

To address these issues, this paper introduces **M**ulti-party **C**ollaborative **A**ttention **C**on**tr**o**l** (MCA-Ctrl), a tuning-free method that enables high-quality image customization using both text and complex visual conditions. Specifically, MCA-Ctrl leverages two key operations within the self-attention layer to coordinate multiple parallel diffusion processes and guide the target image generation. This approach allows MCA-Ctrl to capture the content and appearance of specific subjects while maintaining semantic consistency with the conditional input. Additionally, to mitigate subject leakage and confusion issues common in complex visual scenarios, we introduce a Subject Localization Module that extracts precise subject and editable image layers based on user instructions. Extensive quantitative and human evaluation experiments  show that MCA-Ctrl outperforms existing methods in zero-shot image customization, effectively resolving the mentioned issues.

## Run

### 1. install

We implement our method with [diffusers](https://github.com/huggingface/diffusers) code base with similar code structure to [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt).

```base
conda env create -f environment.yml
conda activate mulcolla
```

### 2. run the following command to view the results

To run the synthesis with MCA-Ctrl, single GPU with at least 16 GB VRAM is required. 

The notebook `generation.ipynb` and `edit.ipynb` provide the generation and real editing samples, respectively.

### 3. Local Gradio demo

You can launch the provided Gradio demo locally with

```bash
python demo_en.py
```

![fig1](fig/demo.png)

## Demo of Customization Result

### customization result of MCA-Ctrl

![fig1](fig/fig1.png)

![fig2](fig/fig2.png)

### editing results of MCA-Ctrl in complex visual condition

![complex](fig/complex.png)

### the analysis of $E_{GI}$

![ablation](fig/ablation.png)

## BibTeX
If you find this project useful in your research, please consider cite:

```bibtex
@inproceedings{yang2025mca,
    title={Multi-party Collaborative Attention Control for Image Customization},
    author={Yang, Han and Yang, Chuanguang and Wang, Qiuli and An, Zhulin and Feng, Weilun and Huang, Libo and Xu, Yongjun},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

## Acknowledgements
We thank to [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [FreeCustom](https://github.com/aim-uofa/FreeCustom)