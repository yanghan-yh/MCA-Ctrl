import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .masactrl_utils import AttentionBase, AttentionStore

from torchvision.utils import save_image

from .seq_aligner import get_replacement_mapper

from .ptp_utils import get_time_words_attention_alpha, get_word_inds

MAX_NUM_WORDS = 77

class McaControlReplace(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, prompts, tokenizer, replace_idx, self_replace_steps, start_step=4, end_step=50, start_layer=10, end_layer=16, layer_idx=None, step_idx=None, total_steps=50, mask_s=None, mask_t=None, mask_save_dir=None, model_type="SDXL"):
        super().__init__()
        self.mask_s = mask_s
        self.mask_t = mask_t
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.end_step = end_step
        self.end_layer = end_layer
        self.replace_idx = replace_idx
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(total_steps * self_replace_steps[0]), int(total_steps * self_replace_steps[1])
        print("McaControlReplace at denoising steps: ", self.step_idx)
        print("McaControlReplace at U-Net layers: ", self.layer_idx)
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k_f = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v_f = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        if kwargs.get("is_mask_attn"):
            # print('Mask self-attention......')
            k_t = rearrange(kwargs.get("k_bg"), "(b h) n d -> h (b n) d", h=num_heads)
            v_t = rearrange(kwargs.get("v_bg"), "(b h) n d -> h (b n) d", h=num_heads)

            mask_t = self.mask_t.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
            mask_t = mask_t.flatten()

            mask_r = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask_r = F.interpolate(mask_r, (H, W)).flatten(0).unsqueeze(0)
            mask_r = mask_r.flatten()
            
            sim_fg = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
            sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(sim.dtype).min)

            sim_bg = torch.einsum("h i d, h j d -> h i j", q, k_t) * kwargs.get("scale")
            sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(sim.dtype).min)

            attn_fg = sim_fg.softmax(-1)
            attn_bg = sim_bg.softmax(-1)

            out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_f)
            out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_t)

            out = torch.cat([out_fg, out_bg])

            out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
            
            return out
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v_f):
            v_f = torch.cat([v_f] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v_f)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def attn_batch2(self, q_bg, k_bg, q_fg, k_fg, v_bg, v_fg, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q_bg.shape[0] // num_heads
        H = W = int(np.sqrt(q_bg.shape[1]))
        q_bg = rearrange(q_bg, "(b h) n d -> h (b n) d", h=num_heads)
        k_bg = rearrange(k_bg, "(b h) n d -> h (b n) d", h=num_heads)
        q_fg = rearrange(q_fg, "(b h) n d -> h (b n) d", h=num_heads)
        k_fg = rearrange(k_fg, "(b h) n d -> h (b n) d", h=num_heads)
        v_bg = rearrange(v_bg, "(b h) n d -> h (b n) d", h=num_heads)
        v_fg = rearrange(v_fg, "(b h) n d -> h (b n) d", h=num_heads)

        mask_t = self.mask_t.unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
        mask_t = mask_t.flatten()

        mask_r = self.mask_s.unsqueeze(0).unsqueeze(0)
        mask_r = F.interpolate(mask_r, (H, W)).flatten(0).unsqueeze(0)
        mask_r = mask_r.flatten()
            
        sim_fg = torch.einsum("h i d, h j d -> h i j", q_fg, k_fg) * kwargs.get("scale")
        sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(q_bg.dtype).min)

        sim_bg = torch.einsum("h i d, h j d -> h i j", q_bg, k_bg) * kwargs.get("scale")
        sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(q_bg.dtype).min)

        attn_fg = sim_fg.softmax(-1)
        attn_bg = sim_bg.softmax(-1)

        out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_fg)
        out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_bg)

        out = torch.cat([out_fg, out_bg])

        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))

        if is_cross:
            
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        elif self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:

            # ours
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_condition = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_condition = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_batch2(qu[:num_heads], ku[:num_heads], qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[:num_heads], vu[num_heads:2*num_heads], num_heads, **kwargs)
            out_c_target = self.attn_batch2(qc[:num_heads], kc[:num_heads], qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[:num_heads], vc[num_heads:2*num_heads], num_heads, **kwargs)

            if self.mask_s is not None and self.mask_t is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out = torch.cat([out_u_source, out_u_condition, out_u_target, out_c_source, out_c_condition, out_c_target], dim=0)

            return out

        elif self.cur_step in self.step_idx and self.cur_att_layer // 2 in self.layer_idx:

            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_condition = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_target = self.attn_batch(qu[-num_heads:], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=ku[:num_heads], v_bg=vu[:num_heads], **kwargs)

            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_condition = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], sim[num_heads:2*num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=kc[:num_heads], v_bg=vc[:num_heads], **kwargs)

            if self.mask_s is not None and self.mask_t is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out = torch.cat([out_u_source, out_u_condition, out_u_target, out_c_source, out_c_condition, out_c_target], dim=0)

            return out

        else:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 32 ** 2:
            return attn_base
        else:
            return att_replace

class McaControlGeneration(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, token_idx, self_replace_steps, SAQI_start_step = 0, start_step=4, end_step=50, start_layer=10, end_layer=16, layer_idx=None, step_idx=None, total_steps=50, mask_r=None, mask_save_dir=None, model_type="SD", th=(.3, .3)):
        super().__init__()
        self.mask_r = mask_r
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.end_step = end_step
        self.end_layer = end_layer
        self.token_idx = token_idx
        self.mask_topp = 0.3
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, end_layer))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.num_self_replace = SAQI_start_step, int(total_steps * self_replace_steps)
        print("McaControlGeneration at denoising steps: ", self.step_idx)
        print("McaControlGeneration at U-Net layers: ", self.layer_idx)
        self.cross_attns = []
        self.mask_cur = None
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_r.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_r.png"))

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k_f = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v_f = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        if kwargs.get("is_mask_attn"):
            # print('Mask self-attention......')
            k_t = rearrange(kwargs.get("k_bg"), "(b h) n d -> h (b n) d", h=num_heads)
            v_t = rearrange(kwargs.get("v_bg"), "(b h) n d -> h (b n) d", h=num_heads)


            mask_t = self.mask_cur.unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
            mask_t = mask_t.flatten()

            mask_r = self.mask_r.unsqueeze(0).unsqueeze(0)
            mask_r = F.interpolate(mask_r, (H, W)).flatten(0).unsqueeze(0)
            mask_r = mask_r.flatten()
            
            sim_fg = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
            sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(sim.dtype).min)

            sim_bg = torch.einsum("h i d, h j d -> h i j", q, k_t) * kwargs.get("scale")
            sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(sim.dtype).min)

            attn_fg = sim_fg.softmax(-1)
            attn_bg = sim_bg.softmax(-1)

            out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_f)
            out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_t)

            out = torch.cat([out_fg, out_bg])

            out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
            
            return out
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k_f) * kwargs.get("scale")
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v_f):
            v_f = torch.cat([v_f] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v_f)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def attn_batch2(self, q_bg, k_bg, q_fg, k_fg, v_bg, v_fg, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        B = q_bg.shape[0] // num_heads
        H = W = int(np.sqrt(q_bg.shape[1]))
        q_bg = rearrange(q_bg, "(b h) n d -> h (b n) d", h=num_heads)
        k_bg = rearrange(k_bg, "(b h) n d -> h (b n) d", h=num_heads)
        q_fg = rearrange(q_fg, "(b h) n d -> h (b n) d", h=num_heads)
        k_fg = rearrange(k_fg, "(b h) n d -> h (b n) d", h=num_heads)
        v_bg = rearrange(v_bg, "(b h) n d -> h (b n) d", h=num_heads)
        v_fg = rearrange(v_fg, "(b h) n d -> h (b n) d", h=num_heads)

        mask_t = self.mask_cur.unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, (H, W)).flatten(0).unsqueeze(0)
        mask_t = mask_t.flatten()

        mask_r = self.mask_r.unsqueeze(0).unsqueeze(0)
        mask_r = F.interpolate(mask_r, (H, W)).flatten(0).unsqueeze(0)
        mask_r = mask_r.flatten()
            
        sim_fg = torch.einsum("h i d, h j d -> h i j", q_fg, k_fg) * kwargs.get("scale")
        sim_fg = sim_fg + mask_r.masked_fill(mask_r == 0, torch.finfo(q_bg.dtype).min)

        sim_bg = torch.einsum("h i d, h j d -> h i j", q_bg, k_bg) * kwargs.get("scale")
        sim_bg = sim_bg + mask_t.masked_fill(mask_t == 1, torch.finfo(q_bg.dtype).min)

        attn_fg = sim_fg.softmax(-1)
        attn_bg = sim_bg.softmax(-1)

        out_fg = torch.einsum("h i j, h j d -> h i d", attn_fg, v_fg)
        out_bg = torch.einsum("h i j, h j d -> h i d", attn_bg, v_bg)

        out = torch.cat([out_fg, out_bg])

        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))

        if is_cross:
            # get every steps mask
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        elif self.num_self_replace[0] < self.cur_step < self.num_self_replace[1]:

            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            # get mask
            mask = self.aggregate_cross_attn_map(idx=self.token_idx)  # (2, H, W)
            self.mask_cur = mask[-1]
            self.mask_cur = F.interpolate(self.mask_cur.unsqueeze(0).unsqueeze(0), (H, W)).reshape(-1, 1)
            values, indices = torch.topk(self.mask_cur.view(-1), int(attn.shape[1]*self.mask_topp), sorted=False)
            self.mask_cur[indices]=1

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_batch2(qu[-num_heads:], ku[-num_heads:], qu[:num_heads], ku[:num_heads], vu[-num_heads:], vu[:num_heads], num_heads, **kwargs)
            out_c_target = self.attn_batch2(qc[-num_heads:], kc[-num_heads:], qc[:num_heads], kc[:num_heads], vc[-num_heads:], vc[:num_heads], num_heads, **kwargs)

            if self.mask_cur is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = F.interpolate(self.mask_cur.unsqueeze(0).unsqueeze(0), (H, W))
                mask = mask.reshape(-1, 1)  # (hw, 1)
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)

            return out

        elif self.cur_step in self.step_idx and self.cur_att_layer // 2 in self.layer_idx:
            
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            # get mask
            mask = self.aggregate_cross_attn_map(idx=self.token_idx)  # (2, H, W)
            self.mask_cur = mask[-1]
            self.mask_cur = F.interpolate(self.mask_cur.unsqueeze(0).unsqueeze(0), (H, W)).reshape(-1, 1)
            values, indices = torch.topk(self.mask_cur.view(-1), int(attn.shape[1]*self.mask_topp), sorted=False)
            self.mask_cur[indices]=1

            out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=ku[-num_heads:], v_bg=vu[-num_heads:], **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, k_bg=kc[-num_heads:], v_bg=vc[-num_heads:], **kwargs)

            if self.mask_cur is not None:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)
                
                out_u_target = out_u_target_fg * self.mask_cur + out_u_target_bg * (1 - self.mask_cur)
                out_c_target = out_c_target_fg * self.mask_cur + out_c_target_bg * (1 - self.mask_cur)

            out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)

            return out

        else:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

