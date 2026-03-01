from typing import Dict, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline

# 假设你的其他工具函数都在这
from delta_k_utils import *


def build_pipeline(model_path: str, is_flux: bool = False):
    try:
        device, dtype = "cuda",torch.bfloat16
    except:
        device, dtype = "cpu",torch.bfloat16
    
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    return pipe.to(device)

def run_diffusion_wrapper(gpu_id, model_path, prompt, steps, seed, model_type, return_dict, key):
    """
    这个包装器运行在子进程中
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(device) # 物理锁定 GPU

        sub_cap = BaseCrossAttentionCapture(model_type=model_type)

        result = run_diffusion_once(
            model_path, 
            prompt, 
            steps=steps, 
            seed=seed, 
            modify=None, 
            active_steps=[], 
            layer_paths=None, 
            attn_cap=sub_cap,
            # device=device # 如果函数支持
        )
    
        # 将结果存入共享字典
        return_dict[key] = result
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()

def run_diffusion_once(
    model_path: str,
    prompt: str,
    steps: int = 30,
    seed: int = 42,
    modify: Optional[Dict] = None,
    active_steps: Optional[List[int]] = None,
    layer_paths: Optional[List[str]] = None,
    attn_cap = None
) -> Tuple["PIL.Image.Image", Dict, Dict[int, Dict]]:
    

    # [架构感知 2]：动态默认层路径
    if layer_paths is None:
        layer_paths = [attn_cap.general_layer_prefix+"."]

    if modify is not None:
        for key in ["q", "k", "v"]:
            modify.setdefault(key, {"signal": False})
            modify[key].setdefault("layer_paths", layer_paths)
            
    # [架构感知 3]：携带模型类型去构建 Pipeline (确保精度正确)
    pipe = build_pipeline(model_path)
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", "cuda")
    
    # [架构感知 4]：动态选取挂载的子网络模型
    target_model = getattr(pipe, attn_cap.net)
    
    # 执行挂载
    attn_cap.attach(target_model, modify=modify, active_steps=active_steps or [])
    callback = StepCallback(attn_cap, total_steps=steps)
    
    # [架构感知 5]：动态请求 Pipeline 的回调张量
    tensor_inputs = attn_cap.tensor_inputs
    
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=tensor_inputs,
        generator=torch.Generator(device).manual_seed(seed),
    )
    
    # 提取快照并卸载 Hook
    attn_snapshot = attn_cap.take(clear=True)
    attn_cap.remove()
    
    # 释放显存，防止多次调用 OOM
    del pipe
    torch.cuda.empty_cache()
    
    return result.images[0], attn_snapshot, callback.steps
