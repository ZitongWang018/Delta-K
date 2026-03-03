from typing import Dict, List, Optional

import torch
import os

from delta_k_core import run_diffusion_once,run_diffusion_wrapper
from delta_k_scheduler import get_schedule
from delta_k_utils import *
import multiprocessing as mp
from torch.cuda import device_count

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
    torch.cuda.empty_cache()
    schedule_config = schedule_config or {}
    if qwen_api_key:
        import os
        os.environ["DASHSCOPE_API_KEY"] = qwen_api_key

    if attn_cap is None:
        raise ValueError("必须传入 attn_cap 实例以感知当前模型架构")

    # [架构路由 1]：动态层前缀
    target_layer_prefix = attn_cap.target_layer_prefix
    general_layer_prefix = attn_cap.general_layer_prefix

    # 注意：这里的 run_diffusion_once 需要确保能够接收并使用我们传入的 attn_cap
    # img_baseline, _, steps_record = run_diffusion_once(
    #     model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap
    # )
    img_baseline, _, steps_record = run_diffusion_once(
        model_path, prompt, steps=max(int(steps/4),8), seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap, record_step=1
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

    tokenizers = attn_cap.load_tokenizers(model_path)
    
    # 3. 调用新的映射函数，传入 model_type
    idx_map = map_concepts_to_indices(
        tokenizers=tokenizers, 
        prompt=prompt, 
        concepts=concepts, 
        model_type=attn_cap.model_type # 使用 Capture 对象中存储的类型
    )

    present, missing = analyze_present_missing(img_baseline, prompt, top_k=6)
    # img_baseline, _, _ = run_diffusion_once(
    #         model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap
    #     )
    # img_baseline.save(f"{prompt}_first_generate.png")
    if not missing and schedule != "baseline":
        print('already succeed, regenerating...')
        img_baseline, _, _ = run_diffusion_once(
            model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap,record_step=0
        )
        return img_baseline

    # [架构路由 3]：动态遮罩占位符 (SDXL 用 <|endoftext|>, FLUX 的 T5 用 <pad>)
    mask_placeholder = attn_cap.mask_placeholder
    prompt_masked = mask_prompt_with_missing(prompt, missing, placeholder=mask_placeholder)
    
    # 将 attn_cap 传给特征提取函数
    target_step_max=schedule_config.get("mean_steps", 10)
    k_delta, attn_pos, attn_neg = _collect_k_delta(model_path, prompt, prompt_masked, steps, seed, attn_cap, target_step_max,record_step=target_step_max)
    
    k_present = compute_concept_k_mean(
        steps_record,
        layer_names,
        gather_indices(idx_map, concepts, exclude=missing),
    )
    k_missing = compute_concept_k_mean(steps_record, layer_names, gather_indices(idx_map, missing))

    present_indices = gather_indices(idx_map, present)







#     missing_indices = gather_indices(idx_map, missing)

#    # ... 前文代码保持不变 (计算 indices) ...

#     # 1. 计算统计量 (包含 step_wise)
#     present_stats = calculate_token_stats(steps_record, layer_target, present_indices, "present")
#     missing_stats = calculate_token_stats(steps_record, layer_target, missing_indices, "missing")

#     # 2. 计算分步 Gap Ratio (分析在哪一步开始丢掉概念)
#     step_ids = sorted(present_stats["step_wise"].keys())
#     gap_series = {}
#     for sid in step_ids:
#         p_m = present_stats["step_wise"][sid]["mean"]
#         m_m = missing_stats["step_wise"].get(sid, {}).get("mean", 0.0)
#         gap_series[sid] = m_m / p_m if p_m > 0 else 0.0

#     # 3. 构造深度日志
#     log_entry = {
#         "metadata": {
#             "prompt": prompt,
#             "missing": missing,
#             "present": present,
#             "seed": seed,
#             "steps": steps,
#             "model": attn_cap.__class__.__name__ # 记录是 Flux 还是 SDXL
#         },
#         "overall_stats": {
#             "present": {k:v for k,v in present_stats.items() if k != "step_wise"},
#             "missing": {k:v for k,v in missing_stats.items() if k != "step_wise"},
#         },
#         "temporal_flow": {
#             "present_steps": present_stats["step_wise"],
#             "missing_steps": missing_stats["step_wise"],
#             "gap_series": gap_series
#         }
#     }

#     # 4. 增强的文件保存逻辑
#     log_file = "logs/detailed_attention_flow_sd3.5.jsonl"
#     import os
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

#     return img_baseline










    
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
    final_attn_cap=BaseCrossAttentionCapture(attn_cap.model_type)

    img_final, _, _ = run_diffusion_once(
        model_path,
        prompt,
        steps=steps,
        seed=seed,
        modify=modify,
        active_steps=active_steps,
        layer_paths=[general_layer_prefix], # 动态设定宏观路径
        attn_cap=final_attn_cap,
        record_step=0
    )
    return img_final


def _collect_k_delta(model_path: str, prompt_pos: str, prompt_neg: str, steps: int, seed: int, attn_cap, target_step_max,record_step):
    if device_count() < 2:
        raise RuntimeError("需要至少 2 个 GPU 才能并行运行")
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        # 2. 通过这个 ctx 创建 Manager 和 Process
        return_dict = manager.dict()
        # 定义两个进程，分别绑定 GPU 0 和 GPU 1
        p1 = ctx.Process(
            target=run_diffusion_wrapper, 
            args=(1, model_path, prompt_pos, steps, seed, attn_cap.model_type, return_dict, target_step_max,record_step, "pos")
        )
        p2 = ctx.Process(
            target=run_diffusion_wrapper, 
            args=(2, model_path, prompt_neg, steps, seed, attn_cap.model_type, return_dict, target_step_max,record_step, "neg")
    )
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # 从返回字典中提取结果
        _, attn_pos, step_pos = return_dict["pos"]
        _, attn_neg, step_neg = return_dict["neg"]

    k_mean_pos = collect_k_mean(attn_pos["k"])
    k_mean_neg = collect_k_mean(attn_neg["k"])

    def _to_list(step_dict):
        return [step_dict[idx] for idx in sorted(step_dict.keys())]
    return k_mean_pos - k_mean_neg, _to_list(step_pos), _to_list(step_neg)