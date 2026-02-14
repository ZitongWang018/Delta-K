from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import Adam


def get_schedule(
    name: str,
    *,
    steps: int,
    latent_var: List[float],
    k_present: float,
    k_missing: float,
    mean_targets: List[Dict],
    attn_pos: List[Dict],
    attn_neg: List[Dict],
    indices: List[int],
    config: Dict,
    attn_cap=None,  # [新增参数] 传入 BaseCrossAttentionCapture 的实例
):
    if name == "baseline":
        return []
    if name == "linear_large":
        return build_linear_schedule(
            steps,
            smax=config.get("linear_smax", 0.004),
            t_lin=config.get("linear_T_lin", 12),
        )
    if name == "vlm_gap_triangle":
        return build_vlm_gap_triangle(
            latent_var,
            k_present,
            k_missing,
            smax=config.get("triangle_smax", 0.060),
            win_ratio=config.get("triangle_win_ratio", 0.40),
            alpha_floor=config.get("triangle_alpha_floor", 0.06),
        )
    if name == "vlm_gap_plateau":
        return build_vlm_gap_plateau(
            latent_var,
            k_present,
            k_missing,
            smax=config.get("plateau_smax", 0.060),
            t_plateau=config.get("plateau_t_plateau", 8),
            t_fall=config.get("plateau_t_fall", 20),
            alpha_floor=config.get("plateau_alpha_floor", 0.06),
        )
    if name == "vlm_gap_burst":
        return build_vlm_gap_burst(
            latent_var,
            k_present,
            k_missing,
            smax=config.get("burst_smax", 0.060),
            tau=config.get("burst_tau", 4.0),
            alpha_floor=config.get("burst_alpha_floor", 0.06),
        )
    if name == "mean_of_concept":
        return build_mean_of_concept_schedule(
            target_stats=mean_targets,
            attn_pos=attn_pos,
            attn_neg=attn_neg,
            indices=indices,
            max_steps=config.get("mean_steps", 10),
            attn_cap=attn_cap, # [新增传递] 
            strength_limit=config.get("mean_strength", 0.03),
        )
    raise ValueError(f"未知调度：{name}")

def build_linear_schedule(steps: int, smax: float = 0.02, t_lin: int = 16) -> List[float]:
    schedule = [0.0] * steps
    for idx in range(min(steps, t_lin)):
        schedule[idx] = smax * (1.0 - idx / float(max(1, t_lin)))
    return schedule

def _variance_window(latent_var: List[float]) -> Tuple[np.ndarray, int]:
    values = np.asarray(latent_var, dtype=float)
    total = len(values)
    values = values / max(values[0], 1e-12)
    diff = np.zeros(total, dtype=float)
    for idx in range(1, total):
        diff[idx] = max(values[idx - 1] - values[idx], 0.0)
    peak = diff.max()
    if peak <= 0:
        return np.ones(total, dtype=float), max(1, total // 4)
    return diff / peak, int(np.argmax(diff))


def _compute_alpha(k_present: float, k_missing: float, alpha_floor: float) -> float:
    denom = max(k_present + k_missing, 1e-12)
    alpha = max(alpha_floor, max(0.0, min(1.0, (k_present - k_missing) / denom)))
    return alpha


def build_vlm_gap_triangle(
    latent_var: List[float],
    k_present: float,
    k_missing: float,
    smax: float = 0.025,
    win_ratio: float = 0.45,
    alpha_floor: float = 0.10,
) -> List[float]:
    window, peak_idx = _variance_window(latent_var)
    total = len(window)
    width = max(2, int(round(win_ratio * total)))
    triangle = np.zeros(total, dtype=float)
    for idx in range(total):
        distance = abs(idx - peak_idx)
        if distance <= width:
            triangle[idx] = 1.0 - distance / float(width)
    alpha = _compute_alpha(k_present, k_missing, alpha_floor)
    return (smax * alpha * triangle * window).tolist()


def build_vlm_gap_plateau(
    latent_var: List[float],
    k_present: float,
    k_missing: float,
    smax: float = 0.028,
    t_plateau: int = 10,
    t_fall: int = 24,
    alpha_floor: float = 0.10,
) -> List[float]:
    window, _ = _variance_window(latent_var)
    total = len(window)
    schedule = np.zeros(total, dtype=float)
    for idx in range(min(total, t_plateau)):
        schedule[idx] = 1.0
    for idx in range(t_plateau, min(total, t_fall)):
        schedule[idx] = 1.0 - (idx - t_plateau) / float(max(1, t_fall - t_plateau))
    alpha = _compute_alpha(k_present, k_missing, alpha_floor)
    return (smax * alpha * schedule * window).tolist()


def build_vlm_gap_burst(
    latent_var: List[float],
    k_present: float,
    k_missing: float,
    smax: float = 0.030,
    tau: float = 4.0,
    alpha_floor: float = 0.10,
) -> List[float]:
    window, _ = _variance_window(latent_var)
    total = len(window)
    time = np.arange(total, dtype=float)
    burst = np.exp(-time / max(1e-6, tau))
    alpha = _compute_alpha(k_present, k_missing, alpha_floor)
    return (smax * alpha * burst * window).tolist()

def build_mean_of_concept_schedule(
    target_stats: List[Dict],
    attn_pos: List[Dict],
    attn_neg: List[Dict],
    indices: List[int],
    max_steps: int,
    attn_cap,  # [新增参数] 接收捕获实例以获取动态的方法和模型属性
    strength_limit: float = 0.03,
) -> List[float]:
    schedule = []
    
    # [核心修改 1] 动态匹配层名前缀
    # SDXL 通常通过干预 down_blocks.1 来控制早期概念布局
    # FLUX 则干预早中期的 transformer_blocks
    layer_filter = "transformer_blocks" if attn_cap.model_type == "flux" else "down_blocks.1"

    for step in range(min(len(attn_pos), max_steps)):
        if step >= len(attn_neg) or step >= len(target_stats):
            break
            
        layer_names = [
            name for name in attn_pos[step]["attention_weights"].keys() if layer_filter in name
        ]
        
        if not layer_names:
            schedule.append(0.0)
            continue

        def objective(alpha: torch.Tensor):
            total = 0.0
            for name in layer_names:
                a_mean = target_stats[step][name]
                
                # [核心修改 2] 调用实例的多态 attention_qk 方法
                attn = attn_cap.attention_qk(
                    attn_pos[step]["q_record"][name],
                    alpha * (attn_pos[step]["k_record"][name] - attn_neg[step]["k_record"][name]),
                )
                
                attn = attn[0, :, indices]
                expanded = torch.broadcast_to(a_mean.unsqueeze(1), attn.shape)
                total = total + ((expanded - attn) ** 2).sum()
            return total

        alpha = torch.tensor([0.05], requires_grad=True)
        optimizer = Adam([alpha], lr=1e-3)
        for _ in range(100):
            optimizer.zero_grad()
            loss_val = objective(alpha)
            grad = torch.autograd.grad(loss_val, alpha, create_graph=True)[0]
            (grad ** 2).backward()
            optimizer.step()
            
        value = float(alpha.item())
        value = max(-strength_limit, min(strength_limit, value))
        schedule.append(value)
        
    return schedule
