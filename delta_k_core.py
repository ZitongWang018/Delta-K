from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline

# 假设你的其他工具函数都在这
from delta_k_utils import *


def build_pipeline(model_path: str):
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

def run_diffusion_once(
    model_path: str,
    prompt: Union[str, List[str]],
    steps: int = 30,
    seed: int = 42,
    guidance_scale: bool = True,
    modify: Optional[Dict] = None,
    active_steps: Optional[List[int]] = None,
    layer_paths: Optional[List[str]] = None,
    attn_cap = None,
    target_step_max = -1,
    record_step = 50,
    pipe: Optional[DiffusionPipeline] = None
) -> Tuple["PIL.Image.Image", Dict, Dict[int, Dict]]:
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    is_internal_pipe = False
    # [架构感知 2]：动态默认层路径
    if layer_paths is None:
        layer_paths = [attn_cap.general_layer_prefix+"."]

    if modify is not None:
        for key in ["q", "k", "v"]:
            modify.setdefault(key, {"signal": False})
            modify[key].setdefault("layer_paths", layer_paths)
            
    # [架构感知 3]：携带模型类型去构建 Pipeline (确保精度正确)
    if pipe is None:
        pipe = build_pipeline(model_path)
        is_internal_pipe = True
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", "cuda")
    # [架构感知 4]：动态选取挂载的子网络模型
    target_model = getattr(pipe, attn_cap.net)
    
    # 执行挂载
    attn_cap.attach(target_model, modify=modify, active_steps=active_steps or [])
    callback = StepCallback(attn_cap, total_steps=steps, break_step=target_step_max,record_step=record_step,batch_size=batch_size)
    
    # [架构感知 5]：动态请求 Pipeline 的回调张量
    tensor_inputs = attn_cap.tensor_inputs
    try:
        if not guidance_scale:
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale = 1.0,
                output_type="pil",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=tensor_inputs,
                generator=torch.Generator(device).manual_seed(seed),
            )
        else:
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                output_type="pil",
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=tensor_inputs,
                generator=torch.Generator(device).manual_seed(seed),
            )
        return_image = result.images
    except StepInterruptException:
        # 捕获到中断信号，此时 pipe 已经停止运行
        print(f"Inference stopped early at step {target_step_max}")
        return_image = None
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise e # 继续抛出异常
    finally:
        # [核心清理逻辑]：无论成功与否，必须执行
        callback.finalize()
        # 提取快照并卸载 Hook (确保模型恢复原样)
        attn_snapshot = attn_cap.take(clear=True)
        attn_cap.remove() 
        
        # 只有当 pipe 是本函数内部创建时，才销毁它
        if is_internal_pipe:
            del pipe
            torch.cuda.empty_cache()
    
    return return_image, attn_snapshot, callback.steps
