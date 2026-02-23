import base64
import io
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
from PIL import Image

try:
    from transformers import CLIPTokenizer, T5TokenizerFast
except Exception:
    CLIPTokenizer = None
    T5TokenizerFast = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ==========================================
# 1. 核心架构：面向对象的 Attention Capture
# ==========================================

class BaseCrossAttentionCapture:
    """
    [架构无关] 注意力捕获基础父类。
    处理数据存储、Hook 管理、特征注入等通用逻辑。
    """
    def __new__(cls, model_type: str, *args, **kwargs):
        if cls is BaseCrossAttentionCapture:
            if model_type not in MODEL_TO_CLASS:
                raise ValueError(f"不支持的模型类型: '{model_type}'。支持的类型有: {list(MODEL_TO_CLASS.keys())}")
            
            target_class = MODEL_TO_CLASS[model_type]
            instance = super().__new__(target_class)
            return instance
            
        return super().__new__(cls)

    def __init__(self, model_type: str) -> None:
        if getattr(self, "_initialized", False):
            return
            
        self.q: Dict[str, Dict[str, torch.Tensor]] = {}
        self.k: Dict[str, Dict[str, torch.Tensor]] = {}
        self.v: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.step = 0
        self.model_type = model_type
        self.target_layer_prefix = self._get_target_layer_prefix(model_type)
        self.general_layer_prefix = self._get_general_layer_prefix(model_type)
        self.mask_placeholder = self._get_mask_placeholder(model_type)
        self.net = self._get_net(model_type)
        self.tensor_inputs = self._get_tensor_inputs(model_type)
        self._initialized = True
    
    def _get_tensor_inputs(self, model_type):
        if model_type in ['flux', 'sd3']:
            return ["latents", "prompt_embeds"]
        if model_type == 'sdxl':
            return ["latents"]

    def _get_net(self, model_type):
        if model_type in ['flux', 'sd3']:
            return 'transformer'
        if model_type == 'sdxl':
            return 'unet'

    def _get_mask_placeholder(self, model_type):
        if model_type in ['flux', 'sd3']:
            return "<pad>"
        if model_type == 'sdxl':
            return "<|endoftext|>"

    def _get_target_layer_prefix(self, model_type):
        # 修复：SD3.5 同样使用 transformer_blocks
        if model_type in ['flux', 'sd3']:
            return "transformer_blocks"
        if model_type == 'sdxl':
            return "down_blocks.1"

    def _get_general_layer_prefix(self, model_type):
        # 修复：SD3.5 同样使用 transformer_blocks
        if model_type in ['flux', 'sd3']:
            return "transformer_blocks"
        if model_type == 'sdxl':
            return "down_blocks"

    def _make_hook(self, layer_path: str, key_type: str, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None):
        def hook(module, inputs, outputs):
            tensor = inputs[0]
            self._maybe_inject(tensor, layer_path, key_type, modify, active_steps)
            
            target_dict = getattr(self, key_type)
            target_dict[layer_path] = {"input": tensor.detach().cpu(), "output": outputs.detach().cpu()}
        return hook

    def _maybe_inject(self, tensor, layer_path, key, modify, active_steps):
        if not modify or key not in modify or not modify[key].get("signal", False) or not active_steps or self.step not in active_steps:
            return
        config = modify[key]
        if not any(item in layer_path for item in config.get("layer_paths", [])):
            return
        if "strength_schedule" in config:
            idx = max(0, min(self.step - 1, len(config["strength_schedule"]) - 1))
            strength = float(config["strength_schedule"][idx])
        else:
            strength = float(config.get("strength", 0.0))
        delta = config.get("value")
        if delta is None or strength == 0.0:
            return
        tensor.data.add_(delta.to(tensor.device) * strength)

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def take(self, clear: bool = True):
        payload = {"q": self.q.copy(), "k": self.k.copy(), "v": self.v.copy()}
        if clear:
            self.q.clear()
            self.k.clear()
            self.v.clear()
        return payload

    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        raise NotImplementedError()

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def load_tokenizers(self, model_path: str):
        raise NotImplementedError()


class SDXLCapture(BaseCrossAttentionCapture):
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        active_steps = active_steps or []
        for bidx, down_block in enumerate(model.down_blocks):
            if not hasattr(down_block, "attentions"):
                continue
            for aidx, attention in enumerate(down_block.attentions):
                for tidx, transformer in enumerate(attention.transformer_blocks):
                    attn2 = transformer.attn2
                    base = f"down_blocks.{bidx}.attentions.{aidx}.transformer_blocks.{tidx}.attn2"
                    
                    self.hooks.append(attn2.to_q.register_forward_hook(self._make_hook(f"{base}.to_q", "q", modify, active_steps)))
                    self.hooks.append(attn2.to_k.register_forward_hook(self._make_hook(f"{base}.to_k", "k", modify, active_steps)))
                    self.hooks.append(attn2.to_v.register_forward_hook(self._make_hook(f"{base}.to_v", "v", modify, active_steps)))

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    def load_tokenizers(self, model_path: str):
        if CLIPTokenizer is None:
            return None, None
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
        return tok1, tok2


class FluxCapture(BaseCrossAttentionCapture):
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        active_steps = active_steps or []
        for tidx, block in enumerate(model.transformer_blocks):
            attn = block.attn
            base = f"transformer_blocks.{tidx}.attn"
            
            self.hooks.append(attn.to_q.register_forward_hook(self._make_hook(f"{base}.to_q", "q", modify, active_steps)))
            self.hooks.append(attn.add_k_proj.register_forward_hook(self._make_hook(f"{base}.add_k_proj", "k", modify, active_steps)))
            self.hooks.append(attn.add_v_proj.register_forward_hook(self._make_hook(f"{base}.add_v_proj", "v", modify, active_steps)))

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor, head_dim: int = 64) -> torch.Tensor:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        
        b, sq, d = q.shape
        _, sk, _ = k.shape
        heads = d // head_dim 
        
        q = q.view(b, sq, heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        
        probs = probs.mean(dim=1) 
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    def load_tokenizers(self, model_path: str):
        if CLIPTokenizer is None:
            return None, None
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        if T5TokenizerFast is None:
            raise ImportError("加载 FLUX 需要 T5TokenizerFast。")
        tok2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
        return tok1, tok2


class SD3Capture(BaseCrossAttentionCapture):
    """
    [SD3.5 适配] Stable Diffusion 3 / 3.5 模型的注意力捕获子类。
    """
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        active_steps = active_steps or []
        
        # 修复：经过上一步探测，SD3.5 在 Diffusers 中的变量名依然是 transformer_blocks
        for tidx, block in enumerate(model.transformer_blocks):
            attn = block.attn
            base = f"transformer_blocks.{tidx}.attn"
            
            self.hooks.append(attn.to_q.register_forward_hook(self._make_hook(f"{base}.to_q", "q", modify, active_steps)))
            self.hooks.append(attn.add_k_proj.register_forward_hook(self._make_hook(f"{base}.add_k_proj", "k", modify, active_steps)))
            self.hooks.append(attn.add_v_proj.register_forward_hook(self._make_hook(f"{base}.add_v_proj", "v", modify, active_steps)))

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor, head_dim: int = 64) -> torch.Tensor:
        # SD3 多头注意力计算 (与 FLUX 计算方式相同，动态推断 Heads)
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        
        b, sq, d = q.shape
        _, sk, _ = k.shape
        heads = d // head_dim 
        
        q = q.view(b, sq, heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        
        probs = probs.mean(dim=1) 
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    def load_tokenizers(self, model_path: str):
        if CLIPTokenizer is None:
            return None, None
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
        return tok1, tok2


MODEL_TO_CLASS = {
    'flux': FluxCapture,
    'sdxl': SDXLCapture,
    'sd3': SD3Capture
}

# ==========================================
# 2. 回调函数改造
# ==========================================

class StepCallback:
    def __init__(self, attn_cap: BaseCrossAttentionCapture, total_steps: int = 30):
        self.attn_cap = attn_cap
        self.total_steps = total_steps
        self.steps: Dict[int, Dict] = {}

    def __call__(self, pipe, step, timestep, kwargs):
        self.attn_cap.step = step + 1
        records = self.attn_cap.take(clear=False)
        weights = {}
        q_record = {}
        k_record = {}
        
        # 【修改点 1】：直接遍历 q 字典的键，而不是用列表索引
        for q_key, q_data in records["q"].items():
            # 获取当前层的基础名字 (例如 "transformer_blocks.0.attn")
            base_name = q_key.replace(".to_q", "") 
            
            # 【修改点 2】：动态探测对应的 K 层名字，兼容不同架构
            if f"{base_name}.to_k" in records["k"]:
                k_key = f"{base_name}.to_k"          # 适配 SDXL / SD1.5
            elif f"{base_name}.add_k_proj" in records["k"]:
                k_key = f"{base_name}.add_k_proj"    # 适配 FLUX / SD3.5
            else:
                continue # 如果找不到对应的 K，为了安全直接跳过
                
            q = q_data["output"]
            k = records["k"][k_key]["output"]
            
            # 兼容 CFG(B=2) 和 Non-CFG(B=1)
            q_cond = q[-1:]
            k_cond = k[-1:]
            
            # 通过多态调用各自架构的 attention 计算
            weights[base_name] = self.attn_cap.attention_qk(q_cond, k_cond)
            q_record[base_name] = q_cond
            k_record[base_name] = k_cond
            
        latents = kwargs.get("latents", kwargs.get("hidden_states"))
        
        self.steps[step] = {
            "step": step,
            "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
            "attention_weights": weights,
            "q_record": q_record,
            "k_record": k_record,
            "latents": latents.detach().cpu() if latents is not None else None,
        }
        return kwargs

# ==========================================
# 4. 测试与验证 Main 函数
# ==========================================

def run_test(model_type: str):
    """一键跑通测试函数，支持 'flux', 'sdxl', 或 'sd3'"""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "A futuristic city with flying cars, highly detailed"
    steps_to_run = 2

    if model_type == "flux":
        from diffusers import FluxPipeline
        print("\n" + "="*50)
        print("▶ 开始测试 FLUX 架构 (极限省显存 Debug 模式)...")
        # 如果你本地有之前下载好的缓存路径，请替换掉它
        model_path = "black-forest-labs/FLUX.1-dev"
        
        # 【省显存招式 1】：千万不要写 .to(device)!
        pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
            # FLUX 强依赖 T5，所以我们不把它设为 None，而是让它待在内存里
        )
        
        # 【省显存招式 2】：开启模型层面的 CPU 卸载
        # 这会将 FLUX 巨大的 Transformer 和 T5 放在系统内存中，
        # 只有在具体运算某一层时，才把它“搬”进显存，算完立刻“搬”出去。
        pipe.enable_model_cpu_offload() 
        # (同样，如果遇到极端情况还是爆显存，可以使用 pipe.enable_sequential_cpu_offload() )

        attn_cap = FluxCapture("flux")
        target_model = pipe.transformer
        
    elif model_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        print("\n" + "="*50)
        print("▶ 开始测试 SDXL 架构...")
        model_path = "stabilityai/stable-diffusion-xl-base-1.0" 
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        attn_cap = SDXLCapture("sdxl")
        target_model = pipe.unet 

    elif model_type == "sd3":
        from diffusers import StableDiffusion3Pipeline
        print("\n" + "="*50)
        print("▶ 开始测试 SD3.5 架构 (极限省显存 Debug 模式)...")
        model_path = "stabilityai/stable-diffusion-3.5-medium" 
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            text_encoder_3=None, 
            tokenizer_3=None
        )
        
        pipe.enable_model_cpu_offload() 

        attn_cap = SD3Capture("sd3")
        target_model = pipe.transformer
        
    else:
        raise ValueError("不支持的模型类型")

    attn_cap.attach(target_model, active_steps=[1, 2])
    callback = StepCallback(attn_cap, total_steps=steps_to_run)
    
    print(f"正在生成图像，步数: {steps_to_run}")
    try:
        pipe(
            prompt=prompt,
            num_inference_steps=steps_to_run,
            height=256, width=256, 
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=attn_cap.tensor_inputs 
        )
        captured_steps = list(callback.steps.keys())
        if captured_steps:
            step_data = callback.steps[captured_steps[0]]
            layers = list(step_data["attention_weights"].keys())
            print(f"✅ [{model_type.upper()}] 测试成功！")
            print(f"   捕获到的步骤数: {len(captured_steps)}")
            print(f"   捕获到的层级数: {len(layers)}")
            if layers:
                print(f"   示例层级: {layers[0]}")
                print(f"   Attention 矩阵形状: {step_data['attention_weights'][layers[0]].shape}")
    except Exception as e:
        print(f"❌ 运行报错: {e}")
    finally:
        attn_cap.remove()
        del pipe
        torch.cuda.empty_cache()
        print("="*50)

if __name__ == "__main__":
    TEST_FLUX = True
    TEST_SDXL = False
    TEST_SD3 = False

    if TEST_FLUX:
        run_test("flux")
    if TEST_SDXL:
        run_test("sdxl")
    if TEST_SD3:
        run_test("sd3")