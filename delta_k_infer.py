import argparse
import runpy
from pathlib import Path
from typing import List, Optional

from delta_k_pipeline import generate_image_with_schedule

DEFAULT_CONFIG = {
    "model": {"base_path": "/root/autodl-tmp/SDXL1.0"},
    "data": {
        "prompt_file": "/root/autodl-tmp/T2I-CompBench/examples/dataset/color_val.txt",
        "single_prompt": None,
    },
    "inference": {
        "steps": 40,
        "seed": 42,
        "schedule": "mean_of_concept",
        "batch_size": 1,
        "n_iter": 1,
    },
    "output": {"dir": "/root/autodl-tmp/T2I-CompBench/examples"},
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


def parse_args():
    parser = argparse.ArgumentParser(description="Delta-K test-time scaling 推理脚本")
    parser.add_argument("--config_py", type=str, default=None, help="可选：包含 CONFIG/DEFAULT_CONFIG 的 Python 文件")
    parser.add_argument("--schedule", type=str, default=None, help="要使用的调度策略")
    parser.add_argument("--from_file", type=str, default=None, help="包含 prompts 的 txt 文件")
    parser.add_argument("--prompt", type=str, default=None, help="单个 prompt 文本")
    parser.add_argument("--outdir", type=str, default=None, help="输出目录")
    parser.add_argument("--steps", type=int, default=None, help="扩散步数")
    parser.add_argument("--n_iter", type=int, default=None, help="每个 prompt 采样次数")
    parser.add_argument("--batch_size", type=int, default=None, help="每个 prompt 的 batch 大小")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--ckpt", type=str, default=None, help="模型路径")
    parser.add_argument("--qwen_api_key", type=str, default=None, help="用于 present/missing 的 API key")
    return parser.parse_args()


def load_prompts_from_file(path: str, batch_size: int = 1) -> List[List[str]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到 prompt 文件: {file_path}")
    prompts = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip().split("\t")[0]
        prompts.append([prompt] * batch_size)
    return prompts


def load_prompts(prompt: Optional[str], prompt_file: Optional[str], batch_size: int = 1) -> List[List[str]]:
    if prompt_file:
        return load_prompts_from_file(prompt_file, batch_size=batch_size)
    if not prompt:
        raise ValueError("未指定 prompt 或 prompt 文件")
    return [[prompt] * batch_size]


def load_config(path: Optional[str]):
    if not path:
        return DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    data = runpy.run_path(str(cfg_path))
    for key in ("CONFIG", "DEFAULT_CONFIG"):
        if key in data:
            return data[key]
    raise ValueError(f"{cfg_path} 中没有 CONFIG/DEFAULT_CONFIG 变量")


def main():
    args = parse_args()
    cfg = load_config(args.config_py)
    model_path = args.ckpt or cfg["model"]["base_path"]
    steps = args.steps or cfg["inference"]["steps"]
    seed = args.seed or cfg["inference"]["seed"]
    schedule = args.schedule or cfg["inference"]["schedule"]
    n_iter = args.n_iter or cfg["inference"]["n_iter"]
    batch_size = args.batch_size or cfg["inference"]["batch_size"]
    prompt_file = args.from_file or cfg["data"]["prompt_file"]
    prompt_text = args.prompt or cfg["data"]["single_prompt"]
    outdir = args.outdir or cfg["output"]["dir"]
    sched_config = cfg.get("schedulers", {})

    prompts = load_prompts(prompt_text, prompt_file, batch_size=batch_size)
    outdir = Path(outdir)
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    counter = len(list(sample_dir.glob("*.png")))
    errors = 0
    for batch_prompts in prompts:
        for iteration in range(n_iter):
            try:
                image = generate_image_with_schedule(
                    model_path=model_path,
                    prompt=batch_prompts[0],
                    schedule=schedule,
                    steps=steps,
                    seed=seed + iteration,
                    qwen_api_key=args.qwen_api_key,
                    schedule_config=sched_config,
                )
                filename = f"{batch_prompts[0]}_{counter:06d}.png"
                image.save(sample_dir / filename)
                counter += 1
            except Exception as exc:
                errors += 1
                print(f"[WARN] 采样失败：{exc}")
    print(f"完成推理，失败次数：{errors}")


if __name__ == "__main__":
    main()

