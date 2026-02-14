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
        # 只有在直接实例化 BaseCrossAttentionCapture 时，才进行路由分发
        if cls is BaseCrossAttentionCapture:
            # 这里的 MODEL_TO_CLASS 字典定义在文件末尾，确保子类已经加载
            if model_type not in MODEL_TO_CLASS:
                raise ValueError(f"不支持的模型类型: '{model_type}'。支持的类型有: {list(MODEL_TO_CLASS.keys())}")
            
            target_class = MODEL_TO_CLASS[model_type]
            # 创建并返回目标子类的实例
            instance = super().__new__(target_class)
            return instance
            
        # 如果是子类自己被实例化（比如直接调用 FluxCapture()），则走正常流程
        return super().__new__(cls)

    def __init__(self, model_type: str) -> None:
        # 添加一个保护机制，防止 __init__ 被调用两次
        if getattr(self, "_initialized", False):
            return
            
        self.q: Dict[str, Dict[str, torch.Tensor]] = {}
        self.k: Dict[str, Dict[str, torch.Tensor]] = {}
        self.v: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.step = 0
        self.model_type = model_type
        
        self._initialized = True
        

    def _make_hook(self, layer_path: str, key_type: str, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None):
        """通用的 Hook 工厂函数，通过 key_type ('q', 'k', 'v') 决定存储位置"""
        def hook(module, inputs, outputs):
            tensor = inputs[0]
            self._maybe_inject(tensor, layer_path, key_type, modify, active_steps)
            
            target_dict = getattr(self, key_type)
            target_dict[layer_path] = {"input": tensor.detach().cpu(), "output": outputs.detach().cpu()}
        return hook

    def _maybe_inject(self, tensor, layer_path, key, modify, active_steps):
        """通用的特征注入逻辑"""
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
        """卸载所有 Hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def take(self, clear: bool = True):
        """提取本 Step 收集到的所有张量"""
        payload = {"q": self.q.copy(), "k": self.k.copy(), "v": self.v.copy()}
        if clear:
            self.q.clear()
            self.k.clear()
            self.v.clear()
        return payload

    # --------- 需要子类实现的方法 ---------
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        raise NotImplementedError("子类必须实现 attach 方法，以挂载到特定架构的层上。")

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("子类必须实现具体的 Attention 计算逻辑。")

    def load_tokenizers(self, model_path: str):
        raise NotImplementedError("子类必须实现具体的 Attention 计算逻辑。")
class SDXLCapture(BaseCrossAttentionCapture):
    """
    [SDXL 适配] SDXL 模型的注意力捕获子类。
    """
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        # 兼容你的原始逻辑：挂载到 UNet 的 down_blocks
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
        # SDXL 标准注意力计算 (不切分 Head 的近似计算)
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    def load_tokenizers(self, model_path: str):
        """
        动态加载 Tokenizer。
        SDXL: Tokenizer1 (CLIP), Tokenizer2 (CLIP)
        FLUX: Tokenizer1 (CLIP), Tokenizer2 (T5)
        """
        if CLIPTokenizer is None:
            return None, None
        # 第一文本编码器：SDXL 和 FLUX 都是 CLIP
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
        return tok1, tok2

class FluxCapture(BaseCrossAttentionCapture):
    """
    [FLUX 适配] FLUX.1 模型的注意力捕获子类。
    """
    def attach(self, model: Any, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        # 挂载到 Transformer 的双流 transformer_blocks
        active_steps = active_steps or []
        for tidx, block in enumerate(model.transformer_blocks):
            attn = block.attn
            base = f"transformer_blocks.{tidx}.attn"
            
            self.hooks.append(attn.to_q.register_forward_hook(self._make_hook(f"{base}.to_q", "q", modify, active_steps)))
            self.hooks.append(attn.add_k_proj.register_forward_hook(self._make_hook(f"{base}.add_k_proj", "k", modify, active_steps)))
            self.hooks.append(attn.add_v_proj.register_forward_hook(self._make_hook(f"{base}.add_v_proj", "v", modify, active_steps)))

    def attention_qk(self, query: torch.Tensor, key: torch.Tensor, heads: int = 24) -> torch.Tensor:
        # FLUX 多头注意力计算 (先切分 Head，计算后取平均)
        q = query.to(torch.float32)
        k = key.to(torch.float32)
        
        b, sq, d = q.shape
        _, sk, _ = k.shape
        head_dim = d // heads
        
        q = q.view(b, sq, heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        
        probs = probs.mean(dim=1) # 合并所有 Head 的响应
        return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    def load_tokenizers(self, model_path: str):
        if CLIPTokenizer is None:
            return None, None
            
        # 第一文本编码器：SDXL 和 FLUX 都是 CLIP
        tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        
        # 第二文本编码器：根据架构动态分发
        if T5TokenizerFast is None:
            raise ImportError("加载 FLUX 需要 T5TokenizerFast，请确保安装了正确的 transformers 版本。")
        tok2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
            
        return tok1, tok2


MODEL_TO_CLASS = {
    'flux': FluxCapture,
    'sdxl': SDXLCapture
}
# ==========================================
# 2. 回调函数改造
# ==========================================

class StepCallback:
    """通用回调函数，接收任何 BaseCrossAttentionCapture 的子类"""
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
        
        q_keys = list(records["q"].keys())
        k_keys = list(records["k"].keys())
        
        for idx in range(min(len(q_keys), len(k_keys))):
            q = records["q"][q_keys[idx]]["output"]
            k = records["k"][k_keys[idx]]["output"]
            
            # 清理后缀以获取准确层名
            layer_name = q_keys[idx].replace(".to_q", "") 
            
            # 兼容 CFG(B=2) 和 Non-CFG(B=1)
            q_cond = q[-1:]
            k_cond = k[-1:]
            
            # 【核心修改】：通过多态调用各自架构的 attention 计算
            weights[layer_name] = self.attn_cap.attention_qk(q_cond, k_cond)
            q_record[layer_name] = q_cond
            k_record[layer_name] = k_cond
            
        # 兼容不同模型的潜变量命名
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


def map_concepts_to_indices(tok1, tok2, prompt: str, concepts: List[str], max_tokens: int) -> Dict[str, List[int]]:
    if tok1 is None or tok2 is None:
        return {concept: [] for concept in concepts}
    t1 = [_clean_token(x) for x in tokens_from_prompt(tok1, prompt)]
    t2 = [_clean_token(x) for x in tokens_from_prompt(tok2, prompt)]
    total = min(max_tokens, len(t1) + len(t2))
    mapping: Dict[str, List[int]] = {concept: [] for concept in concepts}
    for concept in concepts:
        parts = [_clean_token(x) for x in re.findall(r"[a-z]+", concept)]
        for idx, word in enumerate(t1):
            if idx >= total:
                break
            if any(p and p in word for p in parts):
                mapping[concept].append(idx)
        for idx, word in enumerate(t2):
            real_idx = len(t1) + idx
            if real_idx >= total:
                break
            if any(p and p in word for p in parts):
                mapping[concept].append(real_idx)
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


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{content}"


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def analyze_present_missing(image: Image.Image, prompt: str, top_k: int = 8) -> Tuple[List[str], List[str]]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or OpenAI is None:
        return [], []
    api_base_url = os.getenv("VLM_API_BASE_URL")
    if not api_base_url:
        return [], []
    client = OpenAI(api_key=api_key, base_url=api_base_url)
    data_url = pil_to_data_url(image)
    instruction = f"""
You are a vision QA tool. Compare the image with this text prompt:
PROMPT:
{prompt}

Task:
1) List up to {top_k} concept words or short phrases clearly present.
2) List up to {top_k} tokens missing or under-represented.
Rules:
- Use exact words from PROMPT only.
- Return strict JSON with present_tokens, missing_tokens.
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