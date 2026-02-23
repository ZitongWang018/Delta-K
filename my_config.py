"""
Delta-K 推理配置文件。
模型路径可通过环境变量 DELTA_K_MODEL_PATH 设置，或直接修改下方 base_path。
"""
import os

CONFIG = {
    "model": {
        "base_path": os.environ.get("DELTA_K_MODEL_PATH", "/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"),
        "base_path": os.environ.get("DELTA_K_MODEL_PATH", "/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"),
        "base_path": os.environ.get("DELTA_K_MODEL_PATH", "/data/yulin/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"),
        "model_type": "flux",
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
