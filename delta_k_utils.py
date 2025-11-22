import base64
import io
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

try:
    from transformers import CLIPTokenizer
except Exception:
    CLIPTokenizer = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def attention_qk(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)


class CrossAttentionCapture:
    def __init__(self) -> None:
        self.q: Dict[str, Dict[str, torch.Tensor]] = {}
        self.k: Dict[str, Dict[str, torch.Tensor]] = {}
        self.v: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.step = 0

    def _make_hook(self, layer_path: str, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None):
        def hook(module, inputs, outputs):
            tensor = inputs[0]
            if "to_q" in layer_path:
                self._maybe_inject(tensor, layer_path, "q", modify, active_steps)
                self.q[layer_path] = {"input": tensor.detach().cpu(), "output": outputs.detach().cpu()}
            elif "to_k" in layer_path:
                self._maybe_inject(tensor, layer_path, "k", modify, active_steps)
                self.k[layer_path] = {"input": tensor.detach().cpu(), "output": outputs.detach().cpu()}
            elif "to_v" in layer_path:
                self._maybe_inject(tensor, layer_path, "v", modify, active_steps)
                self.v[layer_path] = {"input": tensor.detach().cpu(), "output": outputs.detach().cpu()}

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

    def attach(self, unet, modify: Optional[Dict] = None, active_steps: Optional[List[int]] = None) -> None:
        active_steps = active_steps or []
        for bidx, down_block in enumerate(unet.down_blocks):
            if not hasattr(down_block, "attentions"):
                continue
            for aidx, attention in enumerate(down_block.attentions):
                for tidx, transformer in enumerate(attention.transformer_blocks):
                    attn2 = transformer.attn2
                    base = f"down_blocks.{bidx}.attentions.{aidx}.transformer_blocks.{tidx}.attn2"
                    for name in ["to_q", "to_k", "to_v"]:
                        layer_path = f"{base}.{name}"
                        hook = getattr(attn2, name).register_forward_hook(
                            self._make_hook(layer_path, modify, active_steps)
                        )
                        self.hooks.append(hook)

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


class StepCallback:
    def __init__(self, attn_cap: CrossAttentionCapture, total_steps: int = 30):
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
            weights[q_keys[idx][:-5]] = attention_qk(q, k)[1:, :, :]
            q_record[q_keys[idx][:-5]] = q[1:, :, :]
            k_record[k_keys[idx][:-5]] = k[1:, :, :]
        latents = kwargs.get("latents")
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


def load_tokenizers(model_path: str):
    if CLIPTokenizer is None:
        return None, None
    tok1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    tok2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", local_files_only=True)
    return tok1, tok2


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

