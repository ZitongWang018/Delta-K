from typing import Dict, List, Optional
from pathlib import Path
import torch
import os

from delta_k_core import *
from delta_k_scheduler import get_schedule
from delta_k_utils import *

def _to_list(step_dict):
    return [step_dict[idx] for idx in sorted(step_dict.keys())]


def generate_image_with_schedule(
    model_path: str,
    prompt: str,
    schedule: str = "linear_large",
    *,
    steps: int = 40,
    seed: int = 42,
    qwen_api_key: Optional[str] = None,
    schedule_config: Optional[Dict] = None,
    attn_cap = None,
    main_pipe = None
) -> "PIL.Image.Image":
    torch.cuda.empty_cache()
    schedule_config = schedule_config or {}
    if qwen_api_key:
        import os
        os.environ["DASHSCOPE_API_KEY"] = qwen_api_key

    if attn_cap is None:
        raise ValueError("必须传入 attn_cap 实例以感知当前模型架构")
    if main_pipe is None:
        main_pipe = build_pipeline(model_path)
    try:
        # [架构路由 1]：动态层前缀
        target_layer_prefix = attn_cap.target_layer_prefix
        general_layer_prefix = attn_cap.general_layer_prefix

        # 注意：这里的 run_diffusion_once 需要确保能够接收并使用我们传入的 attn_cap
        # img_baseline, _, steps_record = run_diffusion_once(
        #     model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap
        # )
        img_baseline, _, steps_record = run_diffusion_once(
            model_path, 
            prompt, 
            steps=max(int(steps/2),15), 
            seed=seed, 
            modify=None, 
            active_steps=[], 
            layer_paths=None, 
            attn_cap=attn_cap, 
            record_step=1,
            pipe=main_pipe
        )
        # detect_path = Path(f"./detect/{prompt}_detect.png")
        # detect_path.parent.mkdir(parents=True, exist_ok=True)
        # img_baseline[0].save(detect_path)
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

        # concepts = extract_concepts(prompt)
        sample_layer = layer_names[0] if layer_names else next(iter(steps_record[first_step]["attention_weights"]))
        token_count = steps_record[first_step]["attention_weights"][sample_layer].shape[-1]

        tokenizers = attn_cap.load_tokenizers(model_path)
        present, missing = analyze_present_missing(img_baseline[0], prompt, top_k=6)
        concepts = list(set(present + missing))
        # 3. 调用新的映射函数，传入 model_type
        idx_map = map_concepts_to_indices(
            tokenizers=tokenizers, 
            prompt=prompt, 
            concepts=concepts, 
            model_type=attn_cap.model_type # 使用 Capture 对象中存储的类型
        )

        
        img_baseline, _, _ = run_diffusion_once(
                model_path, 
                prompt, 
                steps=steps, 
                seed=seed, 
                modify=None, 
                active_steps=[], 
                layer_paths=None, 
                attn_cap=attn_cap, 
                record_step=-1,
                pipe=main_pipe
            )
        baseline_path = Path(f"./baseline/{prompt}_first_generate.png")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        img_baseline[0].save(baseline_path)


        if not missing and schedule != "baseline":
            print('already succeed, regenerating...')
            img_baseline, _, _ = run_diffusion_once(
                model_path, prompt, steps=steps, seed=seed, modify=None, active_steps=[], layer_paths=None, attn_cap=attn_cap,record_step=0
            )
            return img_baseline[0]
            return None

        # [架构路由 3]：动态遮罩占位符 (SDXL 用 <|endoftext|>, FLUX 的 T5 用 <pad>)
        mask_placeholder = attn_cap.mask_placeholder
        prompt_masked = mask_prompt_with_missing(prompt, missing, placeholder=mask_placeholder)
        
        # 将 attn_cap 传给特征提取函数
        target_step_max=schedule_config.get("mean_steps", 10)
        k_delta, attn_pos, attn_neg = _collect_k_delta(model_path, prompt, prompt_masked, 
        steps, seed, attn_cap, target_step_max,record_step=target_step_max, pipe=main_pipe)
        k_present = compute_concept_k_mean(
            attn_pos,
            layer_names,
            gather_indices(idx_map, concepts, exclude=missing),
        )
        k_missing = compute_concept_k_mean(attn_pos, layer_names, gather_indices(idx_map, missing))

        present_indices = gather_indices(idx_map, present)
        mean_targets = compute_attention_mean(attn_pos, layer_target, present_indices)
        attn_pos=_to_list(attn_pos)
        attn_neg=_to_list(attn_neg)
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
            return img_baseline[0]
        # if schedule == "baseline" or not schedule_values:
        #     return None
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
            record_step=0,
            pipe=main_pipe
        )
        present_final, missing_final = analyze_present_missing(img_final[0], prompt, top_k=6)
        # if len(missing_final)>=len(missing):
        #     return None
        print('before:',missing)
        print('after:',missing_final)
        return img_final[0]
    finally:
        # [最后清理]：整个流程结束后，销毁主 pipe
        del main_pipe
        torch.cuda.empty_cache()


def _collect_k_delta(
    model_path: str, prompt_pos: str, prompt_neg: str, steps: int, seed: int, 
    attn_cap, target_step_max, record_step, 
    pipe=None # <--- 新增参数，接收主进程的 pipe
):
    
    if pipe is None:
        pipe = build_pipeline(model_path)
    prompts = [prompt_pos, prompt_neg]
    images, combined_attn, combined_steps = run_diffusion_once(
        model_path=model_path, 
        prompt=prompts, 
        steps=steps, 
        seed=seed, 
        guidance_scale=False,
        attn_cap=attn_cap, 
        target_step_max=target_step_max, 
        record_step=record_step,
        pipe=pipe
    )
        
    def extract_step_dict_by_index(steps_dict: dict, idx: int) -> dict:
        """从合并的 Batch 回调字典中剥离出对应 index (0为pos, 1为neg) 的子字典"""
        extracted = {}
        for s, data in steps_dict.items():
            extracted[s] = {
                "step": data["step"],
                "timestep": data["timestep"],
                # 使用 [idx:idx+1] 保持张量的维度数不变 (例如保留 shape 为 (1, S, S))
                "attention_weights": {k: v[idx:idx+1] for k, v in data["attention_weights"].items()},
                "q_record": {k: v[idx:idx+1] for k, v in data["q_record"].items()},
                "k_record": {k: v[idx:idx+1] for k, v in data["k_record"].items()},
                "k_input_record": {k: v[idx:idx+1] for k, v in data["k_input_record"].items()},
                "latents": data["latents"][idx:idx+1] if data["latents"] is not None else None
            }
        return extracted

    def extract_k_mean_by_index(steps_dict: dict, idx: int, step, batch_size: int = 2) -> torch.Tensor:
        """从 k_input_record 中提取全局平均 K，保证空间正确对齐 Hook"""
        all_steps_k_means = []
        
        for data in list(steps_dict.values())[:step]:
            step_k_inputs = []
            # [核心修复]：这里必须遍历 k_input_record，而不是 k_record！
            for layer_name, tensor in data["k_input_record"].items():
                if tensor.shape[0] >= batch_size:
                    cond_tensor = tensor[-batch_size:]
                else:
                    cond_tensor = tensor
                    
                step_k_inputs.append(cond_tensor[idx])
            
            if step_k_inputs:
                step_mean = torch.stack(step_k_inputs, dim=0).mean(dim=0)
                all_steps_k_means.append(step_mean)
                
        if not all_steps_k_means:
            raise ValueError("未捕获到任何步数的 K 输入")
            
        return torch.stack(all_steps_k_means, dim=0).mean(dim=0)

    # ==========================================
    # 3. 拆解数据并返回
    # ==========================================
    # index 0 对应 prompt_pos，index 1 对应 prompt_neg
    step_pos = extract_step_dict_by_index(combined_steps, idx=0)
    step_neg = extract_step_dict_by_index(combined_steps, idx=1)
    
    k_mean_pos = extract_k_mean_by_index(combined_steps, idx=0,step=target_step_max)
    k_mean_neg = extract_k_mean_by_index(combined_steps, idx=1,step=target_step_max)

    # 返回 delta_k，以及独立的 step_pos 和 step_neg
    return k_mean_pos - k_mean_neg, step_pos, step_neg
