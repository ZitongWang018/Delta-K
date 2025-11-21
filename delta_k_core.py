from typing import Dict, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline

from delta_k_utils import CrossAttentionCapture, StepCallback


def resolve_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def build_pipeline(model_path: str):
    device, dtype = resolve_device_dtype()
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
        local_files_only=True,
    )
    return pipe.to(device)


def run_diffusion_once(
    model_path: str,
    prompt: str,
    steps: int = 30,
    seed: int = 42,
    modify: Optional[Dict] = None,
    active_steps: Optional[List[int]] = None,
    layer_paths: Optional[List[str]] = None,
) -> Tuple["PIL.Image.Image", Dict, Dict[int, Dict]]:
    layer_paths = layer_paths or ["down_blocks."]
    if modify is not None:
        for key in ["q", "k", "v"]:
            modify.setdefault(key, {"signal": False})
            modify[key].setdefault("layer_paths", layer_paths)
    pipe = build_pipeline(model_path)
    device = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", "cuda")
    attn_cap = CrossAttentionCapture()
    attn_cap.attach(pipe.unet, modify=modify, active_steps=active_steps or [])
    callback = StepCallback(attn_cap, total_steps=steps)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        output_type="pil",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        generator=torch.Generator(device).manual_seed(seed),
    )
    attn_snapshot = attn_cap.take(clear=True)
    attn_cap.remove()
    return result.images[0], attn_snapshot, callback.steps

