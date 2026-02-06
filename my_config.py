"""
Delta-K 推理配置文件。
模型路径可通过环境变量 DELTA_K_MODEL_PATH 设置，或直接修改下方 base_path。
"""
import os

CONFIG = {
    "model": {
        "base_path": os.environ.get("DELTA_K_MODEL_PATH", "/root/autodl-tmp/model"),
    },
    "data": {
        "prompt_file": None,
        "single_prompt": None,
    },
    "inference": {
        "steps": 40,
        "seed": 42,
        "schedule": "mean_of_concept",
        "batch_size": 1,
        "n_iter": 1,
    },
    "output": {"dir": "output"},
    "schedulers": {
        "linear_smax": 0.004,
        "linear_T_lin": 12,
        "triangle_smax": 0.060,
        "triangle_win_ratio": 0.40,
        "triangle_alpha_floor": 0.06,
        "plateau_smax": 0.060,
        "plateau_t_plateau": 8,
        "plateau_t_fall": 20,
        "plateau_alpha_floor": 0.06,
        "burst_smax": 0.060,
        "burst_tau": 4.0,
        "burst_alpha_floor": 0.06,
        "mean_steps": 10,
        "mean_strength": 0.03,
    },
}
