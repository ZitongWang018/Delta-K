from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import Adam,SGD
import time
import concurrent.futures
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
    attn_cap=None,
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

from torch.optim import Adam
import math

def build_mean_of_concept_schedule(
    target_stats: List[Dict],
    attn_pos: List[Dict],
    attn_neg: List[Dict],
    indices: List[int],
    max_steps: int,
    attn_cap,  # 接收捕获实例，自动获取 head_dim 和 layer_prefix
    strength_limit: float = 0.05,
    max_workers: int=4
) -> List[float]:

    
    # 1. 从 capture 实例获取动态配置
    layer_filter = attn_cap.target_layer_prefix
    head_dim = attn_cap.head_dim
    num_gpus = torch.cuda.device_count()


    # ==========================================
    # 内部辅助函数：复刻 attention_qk 的核心逻辑
    # ==========================================
    def _compute_logits(q, k):
        """计算 Q @ K^T / sqrt(d)，返回"""
        b, sq, d = q.shape
        _, sk, _ = k.shape
        heads = d // head_dim
        
        # 维度 Reshape: (B, S, D) -> (B, Heads, S, HeadDim)
        q = q.view(b, sq, heads, head_dim).transpose(1, 2)
        k = k.view(b, sk, heads, head_dim).transpose(1, 2)
        
        # 计算缩放点积
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    def _logits_to_attn(logits):
        """将 Logits 转换为概率图 (Softmax + Mean Heads)"""
        # 数值稳定性
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)
        # 对 heads 取平均 -> (B, S, S)
        return probs.mean(dim=1)

    # ==========================================
    # 主循环
    # ==========================================
    def optimize_single_step(step: int) -> float:
            
        layer_names = [
            name for name in attn_pos[step]["attention_weights"].keys() 
            if layer_filter in name
        ]
        
        if not layer_names:
            return 0.0
        if num_gpus > 0:
            device = f"cuda:{step % num_gpus}"
        else:
            device = "cpu"
        # ---------------------------------------------------------
        # A. 预计算阶段
        # ---------------------------------------------------------
        # 将 O(N^2*D) 的矩阵乘法提取出来，避免在优化循环中重复计算
        precomputed = []
        
        for name in layer_names:
            # 1. 获取数据 (兼容字典格式 {"input":..., "output":...})
            q_data = attn_pos[step]["q_record"][name]
            k_pos_data = attn_pos[step]["k_record"][name]
            k_neg_data = attn_neg[step]["k_record"][name]
            
            # 提取 output (投影后的张量)
            q = q_data["output"] if isinstance(q_data, dict) else q_data
            k_pos = k_pos_data["output"] if isinstance(k_pos_data, dict) else k_pos_data
            k_neg = k_neg_data["output"] if isinstance(k_neg_data, dict) else k_neg_data
            
            # 转换精度与设备
            q = q.to(torch.float32).to(device)
            k_pos = k_pos.to(torch.float32).to(device)
            k_neg = k_neg.to(torch.float32).to(device)
            
            # 2. 计算 Delta K
            delta_k = k_pos - k_neg
            
            # 3. 计算 Logits (这是最耗时的部分)
            # S_pos = Q @ K_pos.T
            # S_delta = Q @ Delta_K.T
            s_pos = _compute_logits(q, k_pos)
            s_delta = _compute_logits(q, delta_k)
            
            # 4. 获取目标
            s_pos = s_pos.to(torch.bfloat16)
            s_delta = s_delta.to(torch.bfloat16)
            a_mean = target_stats[step][name].to(torch.float32).to(device)
            
            precomputed.append((s_pos, s_delta, a_mean))



        # 初始化 Alpha
        alpha = torch.tensor([0.0], requires_grad=True, device=device)
        optimizer = torch.optim.Adam([alpha], lr=0.003)
        max_time_seconds = 3      # 最大执行时间 (秒)
        patience = 3               # 容忍次数 (连续多少次没改善就停止)
        min_delta = 1e-4 
        best_loss = float('inf')
        no_improve_cnt = 0
        start_time = time.time()
        # 优化循环
        for _ in range(500):
            if time.time() - start_time > max_time_seconds:
                # print(f"Step {step}: 优化超时 ({max_time_seconds}s)，强制中断。") # 取消注释可查看信息
                break

            optimizer.zero_grad()
            current_total_loss = 0.0
            for s_pos, s_delta, a_mean in precomputed:
                # 恢复回 float32 进行高精度 Softmax 和 Loss 计算
                logits = s_pos.to(torch.float32) + alpha * s_delta.to(torch.float32)
                
                attn = _logits_to_attn(logits)
                attn_sel = attn[0, :, indices]
                
                expanded = a_mean.unsqueeze(1).expand_as(attn_sel)
                layer_loss = ((expanded - attn_sel) ** 2).mean()
                
                layer_loss.backward()
                current_total_loss += layer_loss.item()
                
            # 所有层都清算完毕后，统一走一步优化
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=1.0)
            optimizer.step()
            # print(step,alpha)

            
            # [新增] 2. 早停机制逻辑
            if current_total_loss < best_loss - min_delta:
                best_loss = current_total_loss
                no_improve_cnt = 0  # 发生有效改善，重置容忍计数器
            else:
                no_improve_cnt += 1 # 改善微弱或恶化，累加计数器
                
            if no_improve_cnt >= patience:
                # print(f"Step {step}: 连续 {patience} 步未见显著改善，触发早停。") # 取消注释可查看信息
                break
        # 限制范围并记录
        value = float(alpha.item())
        return max(-strength_limit, min(strength_limit, value))
    actual_max_steps = min(len(attn_pos), len(attn_neg), len(target_stats), max_steps)
    
    # 如果没有 step 需要处理，直接返回空
    if actual_max_steps <= 0:
        return []

    # 使用线程池并行执行
    # executor.map 会保证返回的 list 顺序与 range(actual_max_steps) 的顺序完全一致
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        schedule = list(executor.map(optimize_single_step, range(actual_max_steps)))

    return schedule   

