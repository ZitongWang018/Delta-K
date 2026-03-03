import base64
import io
import json
import math
import os
import re
import threading
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from PIL import Image

try:
    from transformers import CLIPTokenizer, T5TokenizerFast
except Exception:
    CLIPTokenizer = None
    T5TokenizerFast = None


from openai import OpenAI


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
        self.head_dim = 64
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

    def _make_hook(self, layer_path: str, key: str, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None):
        def hook(module, inputs, outputs):
            if (not modify or key not in modify or not modify[key].get("signal", False) or not active_steps or self.step not in active_steps)==False:
                tensor = inputs[0]
                self._maybe_inject(tensor, layer_path, key, modify, active_steps)
            target_dict = getattr(self, key)
            target_dict[layer_path] = {"input": inputs[0].detach(), "output": outputs.detach()}
        return hook

    def _maybe_inject(self, tensor, layer_path, key, modify, active_steps):

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
        payload = {
            "q": {k: {'input':v['input'].cpu(),'output':v['output'].cpu()} for k, v in self.q.items()},
            "k": {k: {'input':v['input'].cpu(),'output':v['output'].cpu()} for k, v in self.k.items()},
            "v": {k: {'input':v['input'].cpu(),'output':v['output'].cpu()} for k, v in self.v.items()}
        }
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

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor, head_dim: int = 64) -> torch.Tensor:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        self.head_dim=64
        b, sq, d = q.shape
        _, sk, _ = k.shape
        heads = d // head_dim 
        
        # SDXL 也是多头注意力，需要 reshape
        # (B, S, D) -> (B, Heads, S, HeadDim)
        q = q.view(b, sq, heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        
        # 对 heads 取平均，得到 (B, S, S)
        probs = probs.mean(dim=1) 
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

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        self.head_dim=128
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
        self.head_dim=64
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
        # 【修改点】: SD3 必须加载 3 个 Tokenizer
        if CLIPTokenizer is None:
            return None, None, None
        
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
        
        if T5TokenizerFast is None:
            raise ImportError("加载 SD3 需要 T5TokenizerFast。")
        tok3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3", local_files_only=True)
        
        return tok1, tok2, tok3  # 返回 3 个


MODEL_TO_CLASS = {
    'flux': FluxCapture,
    'sdxl': SDXLCapture,
    'sd3': SD3Capture
}

# ==========================================
# 2. 回调函数改造
# ==========================================
class StepInterruptException(Exception):
    """用于在达到指定步数时强制中断推理的自定义异常"""
    pass
class StepCallback:
    def __init__(self, attn_cap: BaseCrossAttentionCapture, total_steps: int = 30, break_step: int = 10, record_step: int=10):
        self.attn_cap = attn_cap
        self.total_steps = total_steps
        self.steps: Dict[int, Dict] = {}
        self.break_step = break_step
        self.record_step = record_step
        self._threads = [] 
    def _async_process_step(self, step: int, records: Dict, latents: torch.Tensor, timestep):
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
            self.steps[step] = {
            "step": step,
            "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
            "attention_weights": weights,
            "q_record": q_record,
            "k_record": k_record,
            "latents": latents.detach().cpu() if latents is not None else None,
        }
    def __call__(self, pipe, step, timestep, kwargs):
        self.attn_cap.step = step + 1
        if self.break_step>0 and self.attn_cap.step>self.break_step:
            for t in self._threads:
                t.join()
            raise StepInterruptException("Reached target_step_max, interrupting...")
        if self.attn_cap.step >self.record_step:
            return kwargs
        records = self.attn_cap.take(clear=False)

        self.attn_cap.q.clear()
        self.attn_cap.k.clear()
        self.attn_cap.v.clear()
        latents = kwargs.get("latents", kwargs.get("hidden_states"))
        t_val = int(step.item()) if hasattr(step, "item") else int(step)
        
        thread = threading.Thread(
            target=self._async_process_step, 
            args=(step, records, latents,timestep) # 传递 GPU 引用给线程
        )
        thread.start()
        self._threads.append(thread)
        
        return kwargs
    def finalize(self):
        """可选：在所有推理结束后，调用此方法确保所有后台线程已完成"""
        for t in self._threads:
            t.join()
def compute_concept_k_mean(step_data: Dict[int, Dict], layer_names: List[str], indices: List[int]) -> float:
    if not indices:
        return 0.0
    steps = sorted(step_data.keys())
    values = []
    for step in steps:
        for name in layer_names:
            attn = step_data[step]["attention_weights"].get(name)
            if attn is None:
                continue
            mass = attn.sum(dim=1)
            valid_idx = [idx for idx in indices if idx < mass.shape[-1]]
            if not valid_idx:
                continue
            values.append(float(mass.mean(dim=0)[valid_idx].mean().detach().cpu()))
    return float(sum(values) / len(values)) if values else 0.0


def compute_attention_mean(step_data: Dict[int, Dict], layer_names: List[str], indices: List[int]):
    if not indices:
        return []
    steps = sorted(step_data.keys())
    results = []
    for step in steps:
        collected = {}
        for name in layer_names:
            attn = step_data[step]["attention_weights"].get(name)
            # 保留了 down_blocks.2 的跳过逻辑，它对 FLUX 不会产生误伤
            if attn is None or "down_blocks.2" in name:
                continue
            valid_idx = [idx for idx in indices if idx < attn.shape[-1]]
            if not valid_idx:
                continue
            attn_selected = attn[:, :, valid_idx]
            collected[name] = attn_selected.mean(dim=2)[0, :]
        if collected:
            results.append(collected)
    return results


def extract_concepts(prompt: str) -> List[str]:
    text = prompt.lower()
    words = re.findall(r"[a-z]+", text)
    stops = {"their", "this", "that"}
    unigrams = [w for w in words if w not in stops]
    bigrams = []
    for idx in range(len(words) - 1):
        w1, w2 = words[idx], words[idx + 1]
        if w1 in stops or w2 in stops:
            continue
        phrase = f"{w1} {w2}"
        if phrase in text:
            bigrams.append(phrase)
    seen = set()
    results: List[str] = []
    for token in bigrams + unigrams:
        if token not in seen:
            seen.add(token)
            results.append(token)
    return results


def _clean_token(token: str) -> str:
    token = token.lower()
    for bad in ["</w>", "Ġ", "##", "Ċ", "｡", "▁"]:
        token = token.replace(bad, "")
    return re.sub(r"[^a-z]+", "", token)


def tokens_from_prompt(tokenizer, prompt: str) -> List[str]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokens


def map_concepts_to_indices(
    tokenizers: Tuple, 
    prompt: str, 
    concepts: List[str], 
    model_type: str
) -> Dict[str, List[int]]:
    """
    根据模型类型执行正确的索引映射策略。
    
    Args:
        tokenizers: Tokenizer 元组 (数量取决于模型)
        prompt: 原始提示词
        concepts: 概念列表
        model_type: 'sdxl', 'flux', 或 'sd3'
    """
    mapping: Dict[str, List[int]] = {concept: [] for concept in concepts}
    
    def get_clean_tokens(tok):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        return [_clean_token(x) for x in tok.convert_ids_to_tokens(ids)]

    # --- 策略 1: SDXL (索引对齐) ---
    if model_type == 'sdxl':
        tok1, tok2 = tokenizers[0], tokenizers[1]
        t1 = get_clean_tokens(tok1)
        # t2 = get_clean_tokens(tok2) # SDXL中t2与t1索引一一对应，主要用于验证
        
        for concept in concepts:
            parts = [_clean_token(x) for x in re.findall(r"[a-z]+", concept)]
            for idx, word in enumerate(t1):
                # 只要 Tokenizer1 或 Tokenizer2 中有一个包含概念，该索引即有效
                # 注意：这里简化处理，通常 t1 和 t2 的 clean word 是一致的
                if any(p and p in word for p in parts):
                    mapping[concept].append(idx)

    # --- 策略 2: FLUX (仅 T5) ---
    elif model_type == 'flux':
        # FLUX 不使用 tokenizer_1 (CLIP) 的序列索引
        if len(tokenizers) < 2:
            return mapping
        t5_tokens = get_clean_tokens(tokenizers[1])
        
        for concept in concepts:
            parts = [_clean_token(x) for x in re.findall(r"[a-z]+", concept)]
            for idx, word in enumerate(t5_tokens):
                if any(p and p in word for p in parts):
                    mapping[concept].append(idx)

    # --- 策略 3: SD3 (序列拼接: CLIP + T5) ---
    elif model_type == 'sd3':
        if len(tokenizers) < 3:
            return mapping
        
        t_l = get_clean_tokens(tokenizers[0])
        t_g = get_clean_tokens(tokenizers[1])
        t_t5 = get_clean_tokens(tokenizers[2])
        
        # SD3.5 Medium 确认: CLIP 部分固定为 77，T5 部分最大 256
        # 实际逻辑中应使用 tokenizer 的实际长度或模型配置的长度
        clip_len = 77 
        
        for concept in concepts:
            parts = [_clean_token(x) for x in re.findall(r"[a-z]+", concept)]
            
            # 1. CLIP L/G 部分 (索引 0 ~ 76)
            # 逻辑：只要在 L 或 G 中出现，就算作该索引
            for idx in range(clip_len):
                w_l = t_l[idx] if idx < len(t_l) else ""
                w_g = t_g[idx] if idx < len(t_g) else ""
                
                if any(p and (p in w_l or p in w_g) for p in parts):
                    mapping[concept].append(idx)
            
            # 2. T5 部分 (索引 77 ~ 333)
            # 逻辑：如果在 T5 中出现，索引需加上 CLIP 的偏移量
            for idx, word in enumerate(t_t5):
                if any(p and p in word for p in parts):
                    mapping[concept].append(clip_len + idx)

    # 清理重复索引
    for concept in mapping:
        mapping[concept] = sorted(list(set(mapping[concept])))
        
    return mapping


def mask_prompt_with_missing(prompt: str, missing_terms: List[str], placeholder: str = "<pad><pad><pad>") -> str:
    masked = prompt
    for term in sorted(missing_terms, key=len, reverse=True):
        if not term or term not in masked:
            continue
        pieces = term.split(" ")
        replacement = " ".join([placeholder] * len(pieces))
        masked = masked.replace(term, replacement)
    return masked


def collect_k_mean(attn_ref: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    k_inputs = [value["input"] for value in attn_ref.values()]
    if not k_inputs:
        raise ValueError("未捕获到任何 K 输入")
    return torch.stack(k_inputs, dim=0).mean(dim=0)


def gather_indices(idx_map: Dict[str, List[int]], concepts: List[str], exclude: Optional[List[str]] = None) -> List[int]:
    exclude = set(exclude or [])
    indices = sorted({idx for concept in concepts if concept not in exclude for idx in idx_map.get(concept, [])})
    return indices

def calculate_token_stats(step_data: Dict[int, Dict], layer_names: List[str], indices: List[int], label: str):
    if not indices:
        return {f"{label}_{k}": 0.0 for k in ["mean", "max", "std", "var", "cv", "p90", "sparsity"]} | {"step_wise": {}}

    all_raw_values = []
    step_wise_data = {} # 新增：按 step 存储
    steps = sorted(step_data.keys())
    
    for step in steps:
        step_vals = []
        for name in layer_names:
            attn = step_data[step]["attention_weights"].get(name)
            if attn is None: continue
            
            dims_to_reduce = tuple(range(attn.ndim - 1))
            avg_attn = torch.abs(attn).mean(dim=dims_to_reduce)
            
            valid_idx = [i for i in indices if i < avg_attn.shape[0]]
            if valid_idx:
                v = avg_attn[valid_idx].detach().cpu()
                all_raw_values.append(v)
                step_vals.append(v)
        
        # 记录当前 step 的局部统计
        if step_vals:
            step_combined = torch.cat(step_vals)
            step_wise_data[int(step)] = {
                "mean": float(step_combined.mean().item()),
                "max": float(step_combined.max().item()),
                "var": float(step_combined.var().item()) # 考察该步内层间的剧烈程度
            }

    if not all_raw_values:
        return {f"{label}_mean": 0.0, "step_wise": {}}

    combined = torch.cat(all_raw_values)
    mean_v = combined.mean().item()
    std_v = combined.std().item()
    
    stats = {
        f"{label}_mean": mean_v,
        f"{label}_max":  combined.max().item(),
        f"{label}_std":  std_v,
        f"{label}_var":  combined.var().item(), # 全局方差
        f"{label}_cv":   (std_v / mean_v) if mean_v > 0 else 0,
        f"{label}_p90":  torch.quantile(combined, 0.9).item(),
        f"{label}_sparsity": (combined < (mean_v * 0.1)).float().mean().item(),
        "step_wise": step_wise_data # 包含分步明细
    }
    return stats
def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{content}"


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def analyze_present_missing(image: Image.Image, prompt: str, top_k: int = 8) -> Tuple[List[str], List[str]]:
    api_key = "sk-6b8c3387f0374255ab6adb1211382810"
    if not api_key or OpenAI is None:
        print(1)
        return [], []
    api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not api_base_url:
        print(2)
        return [], []
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    data_url = pil_to_data_url(image)
    instruction = f"""
You are a high-precision vision QA tool. Your task is to verify the visual fidelity of an image against a specific text prompt by focusing on concrete entities and their immediate modifiers.

**PROMPT:** {prompt}

**EVALUATION STEPS:**
1. **Entity-Attribute Extraction:** Deconstruct the prompt into specific nouns or short "modifier-noun" phrases (e.g., "vintage clock," "golden retriever," "marble floor"). Ignore broad actions or complex clauses.
2. **Visual Strictness Check:** For each extracted phrase, verify if the visual evidence EXACTLY matches every word in that phrase.
    - *Constraint:* If the prompt specifies a "transparent glass bottle" and the image shows an "opaque plastic bottle," the phrase is marked as missing.
3. **Reasoning:** Evaluate whether the specific physical attributes (color, texture, material, quantity, relative position) described are visibly evident.

**TASK:**
1. **present_tokens:** List up to {top_k} noun-based phrases or entities from the PROMPT where all modifiers are correctly and clearly depicted.
2. **missing_tokens:** List up to {top_k} noun-based phrases or entities from the PROMPT that are absent, have incorrect modifiers (e.g., wrong color/material), or are visually ambiguous.

**STRICT RULES:**
- **Focus:** Extract ONLY nouns or short [Adjective/Modifier] + [Noun] structures. Avoid long sentences or verbs.
- **Exact Match:** Use the EXACT wording from the PROMPT. No paraphrasing.
- **Strict Logic:** Be hyper-critical of modifiers. If a "blue silk tie" is shown as a "blue wool tie," it belongs in `missing_tokens`.
- **Format:** Return a strict JSON object.

**OUTPUT FORMAT:**
{{
  "present_tokens": [],
  "missing_tokens": []
}}
"""
    try:
        response = client.chat.completions.create(
            model=os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus"),
            messages=[
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": instruction}]}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content)
        print(payload)
        raw_present = [t.strip().lower() for t in payload.get("present_tokens", []) if isinstance(t, str)]
        raw_missing = [t.strip().lower() for t in payload.get("missing_tokens", []) if isinstance(t, str)]
    except Exception:
        return [], []
    prompt_words = set(_norm_words(prompt))
    prompt_bigrams = set([" ".join(x) for x in zip(_norm_words(prompt)[:-1], _norm_words(prompt)[1:])])

    def filter_terms(tokens: List[str]) -> List[str]:
        filtered = []
        for token in tokens:
            words = _norm_words(token)
            keep = False
            if len(words) == 1:
                word = words[0] if words else ""
                if word and (word in prompt_words or word.rstrip("s") in prompt_words):
                    keep = True
            else:
                phrase = " ".join(words)
                if phrase in prompt_bigrams or all(word in prompt_words for word in words):
                    keep = True
            if keep and token not in filtered:
                filtered.append(token)
        return filtered[:top_k]

    return filter_terms(raw_present), filter_terms(raw_missing)

# ==========================================
# 4. 测试与验证 Main 函数
# ==========================================

def run_test(model_type: str):
    """一键跑通测试函数，支持 'flux' 或 'sdxl'"""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = "A futuristic city with flying cars, highly detailed"
    steps_to_run = 2

    if model_type == "flux":
        from diffusers import FluxPipeline
        print("\n" + "="*50)
        print("▶ 开始测试 FLUX 架构...")
        model_path = "/data/yulin/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
        
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        attn_cap = FluxCapture()
        target_model = pipe.transformer # FLUX 挂载到 transformer
        
    elif model_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        print("\n" + "="*50)
        print("▶ 开始测试 SDXL 架构...")
        # 填入你本地的 SDXL 路径，此处为占位
        model_path = "stabilityai/stable-diffusion-xl-base-1.0" 
        
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        attn_cap = SDXLCapture()
        target_model = pipe.unet # SDXL 挂载到 unet
    else:
        raise ValueError("不支持的模型类型")

    # 统一接口挂载与执行
    attn_cap.attach(target_model, active_steps=[1, 2])
    callback = StepCallback(attn_cap, total_steps=steps_to_run)
    
    print(f"正在生成图像，步数: {steps_to_run}")
    try:
        pipe(
            prompt=prompt,
            num_inference_steps=steps_to_run,
            height=512, width=512,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"] if model_type=="flux" else ["latents"]
        )
        
        # 验证结果
        captured_steps = list(callback.steps.keys())
        if captured_steps:
            step_data = callback.steps[captured_steps[0]]
            layers = list(step_data["attention_weights"].keys())
            print(f"✅ [{model_type.upper()}] 测试成功！")
            print(f"   捕获到的步骤数: {len(captured_steps)}")
            print(f"   捕获到的层级数: {len(layers)}")
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
    # 你可以修改下面的布尔值来选择测试哪个模型
    TEST_FLUX = False
    TEST_SDXL = True

    if TEST_FLUX:
        run_test("flux")
    
    if TEST_SDXL:
        run_test("sdxl")