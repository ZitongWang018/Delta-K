from typing import Dict, List, Optional

import torch

from delta_k_core import run_diffusion_once
from delta_k_scheduler import get_schedule
from delta_k_utils import *


def generate_image_with_schedule(
    model_path: str,
    prompt: str,
    schedule: str = "linear_large",
    *,
    steps: int = 40,
    seed: int = 42,
    qwen_api_key: Optional[str] = None,
    schedule_config: Optional[Dict] = None,
    attn_cap = None  # 必须传入实例化好的 Capture 对象 (FluxCapture 或 SDXLCapture)
) -> "PIL.Image.Image":
    schedule_config = schedule_config or {}
    if qwen_api_key:
        import os
        os.environ["DASHSCOPE_API_KEY"] = qwen_api_key

    if attn_cap is None:
        raise ValueError("必须传入 attn_cap 实例以感知当前模型架构")

    # [架构路由 1]：动态层前缀
    is_flux = attn_cap.model_type == "flux"
    target_layer_prefix = "transformer_blocks" if is_flux else "down_blocks.1"
    general_layer_prefix = "transformer_blocks" if is_flux else "down_blocks"

    # 注意：这里的 run_diffusion_once 需要确保能够接收并使用我们传入的 attn_cap
    img_baseline, _, steps_record = run_diffusion_once(
        model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap
    )

    step_ids = sorted(steps_record.keys())
    latent_vars = []
    for sid in step_ids:
        latent = steps_record[sid]["latents"]
        value = float(torch.var(latent.to(torch.float32))) if latent is not None else 0.0
        latent_vars.append(value)

    first_step = step_ids[0]
    layer_names = sorted(steps_record[first_step]["attention_weights"].keys())
    
    # [架构路由 2]：动态筛选目标干预层
    layer_target = [name for name in layer_names if target_layer_prefix in name]
    
    if len(layer_names) > 3:
        layer_names = [layer_names[0], layer_names[len(layer_names) // 2], layer_names[-1]]

    concepts = extract_concepts(prompt)
    sample_layer = layer_names[0] if layer_names else next(iter(steps_record[first_step]["attention_weights"]))
    token_count = steps_record[first_step]["attention_weights"][sample_layer].shape[-1]

    tokenizer1, tokenizer2 = attn_cap.load_tokenizers(model_path)
    idx_map = map_concepts_to_indices(tokenizer1, tokenizer2, prompt, concepts, token_count)

    present, missing = analyze_present_missing(img_baseline, prompt, top_k=6)
    if not missing and schedule != "baseline":
        return img_baseline

    # [架构路由 3]：动态遮罩占位符 (SDXL 用 <|endoftext|>, FLUX 的 T5 用 <pad>)
    mask_placeholder = "<pad>" if is_flux else "<|endoftext|>"
    prompt_masked = mask_prompt_with_missing(prompt, missing, placeholder=mask_placeholder)
    
    # 将 attn_cap 传给特征提取函数
    k_delta, attn_pos, attn_neg = _collect_k_delta(model_path, prompt, prompt_masked, steps, seed, attn_cap)
    
    k_present = compute_concept_k_mean(
        steps_record,
        layer_names,
        gather_indices(idx_map, concepts, exclude=missing),
    )
    k_missing = compute_concept_k_mean(steps_record, layer_names, gather_indices(idx_map, missing))

    present_indices = gather_indices(idx_map, present)
    
    # 使用动态匹配的目标层计算均值
    mean_targets = compute_attention_mean(steps_record, layer_target, present_indices)
    
    schedule_values = get_schedule(
        schedule,
        steps=steps,
        latent_var=latent_vars,
        k_present=k_present,
        k_missing=k_missing,
        mean_targets=mean_targets,
        attn_pos=attn_pos,
        attn_neg=attn_neg,
        indices=gather_indices(idx_map, missing),
        config=schedule_config,
        attn_cap=attn_cap, # 传入上一步修改好的参数
    )
    if schedule == "baseline" or not schedule_values:
        return img_baseline

    active_steps = [idx + 1 for idx, _ in enumerate(schedule_values)]
    
    # [架构路由 4]：动态设定特征注入的层路径
    modify = {
        "q": {"signal": False, "layer_paths": []},
        "k": {"signal": True, "value": k_delta, "strength_schedule": schedule_values, "layer_paths": [target_layer_prefix]},
        "v": {"signal": False, "layer_paths": []},
    }
    
    img_final, _, _ = run_diffusion_once(
        model_path,
        prompt,
        steps=steps,
        seed=seed,
        modify=modify,
        active_steps=active_steps,
        layer_paths=[general_layer_prefix], # 动态设定宏观路径
        attn_cap=attn_cap
    )
    return img_final


def _collect_k_delta(model_path: str, prompt_pos: str, prompt_neg: str, steps: int, seed: int, attn_cap):
    # 必须接收并传递 attn_cap
    _, attn_pos, step_pos = run_diffusion_once(model_path, prompt_pos, steps=steps, seed=seed, attn_cap=attn_cap)
    _, attn_neg, step_neg = run_diffusion_once(model_path, prompt_neg, steps=steps, seed=seed, attn_cap=attn_cap)
    
    k_mean_pos = collect_k_mean(attn_pos["k"])
    k_mean_neg = collect_k_mean(attn_neg["k"])

    def _to_list(step_dict: Dict[int, Dict]):
        return [step_dict[idx] for idx in sorted(step_dict.keys())]

    return k_mean_pos - k_mean_neg, _to_list(step_pos), _to_list(step_neg)