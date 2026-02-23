import argparse
import runpy
from pathlib import Path
from typing import List, Optional

from delta_k_pipeline import generate_image_with_schedule
from delta_k_utils import *
DEFAULT_CONFIG = {
    "model": {"base_path": "path/to/your/model"},
    "data": {
        "prompt_file": "path/to/your/prompts.txt",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Delta-K test-time scaling CLI")
    parser.add_argument("--config_py", type=str, default=None, help="Optional Python config exposing CONFIG/DEFAULT_CONFIG")
    parser.add_argument("--schedule", type=str, default=None, help="Sampling schedule name")
    parser.add_argument("--from_file", type=str, default=None, help="Prompt list txt file")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to store outputs")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps")
    parser.add_argument("--n_iter", type=int, default=None, help="Sampling iterations per prompt")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--ckpt", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--qwen_api_key", type=str, default=None, help="API key for present/missing analysis")
    return parser.parse_args()


def load_prompts_from_file(path: str, batch_size: int = 1) -> List[List[str]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f" {file_path}")
    prompts = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip().split("\t")[0]
        prompts.append([prompt] * batch_size)
    return prompts


def load_prompts(prompt: Optional[str], prompt_file: Optional[str], batch_size: int = 1) -> List[List[str]]:
    if prompt_file:
        return load_prompts_from_file(prompt_file, batch_size=batch_size)
    if not prompt:
        raise ValueError("prompt")
    return [[prompt] * batch_size]


def load_config(path: Optional[str]):
    if not path:
        return DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path}")
    data = runpy.run_path(str(cfg_path))
    for key in ("CONFIG", "DEFAULT_CONFIG"):
        if key in data:
            return data[key]
    raise ValueError(f"{cfg_path} ")


def main():
    args = parse_args()
    cfg = load_config(args.config_py)
    model_path = args.ckpt or cfg["model"]["base_path"]
    steps = args.steps or cfg["inference"]["steps"]
    seed = args.seed or cfg["inference"]["seed"]
    schedule = args.schedule or cfg["inference"]["schedule"]
    n_iter = args.n_iter or cfg["inference"]["n_iter"]
    batch_size = args.batch_size or cfg["inference"]["batch_size"]
    # 当传入 --prompt 时优先使用命令行 prompt，不从文件读
    prompt_file = args.from_file if args.from_file else (None if args.prompt else cfg["data"]["prompt_file"])
    prompt_text = args.prompt or cfg["data"]["single_prompt"]
    outdir = args.outdir or cfg["output"]["dir"]
    sched_config = cfg.get("schedulers", {})
    model_type=cfg["model"]["model_type"]

    prompts = load_prompts(prompt_text, prompt_file, batch_size=batch_size)
    outdir = Path(outdir)
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    counter = len(list(sample_dir.glob("*.png")))
    errors = 0
    attn_cap = BaseCrossAttentionCapture(model_type=model_type)
    for batch_prompts in prompts:
        for iteration in range(n_iter):
            # try:
            image = generate_image_with_schedule(
                model_path=model_path,
                prompt=batch_prompts[0],
                schedule=schedule,
                steps=steps,
                seed=seed + iteration,
                qwen_api_key=args.qwen_api_key,
                schedule_config=sched_config,
                attn_cap=attn_cap
            )
            filename = f"{batch_prompts[0]}_{counter:06d}.png"
            image.save(sample_dir / filename)
            counter += 1
            # except Exception as exc:
            #     errors += 1
            #     print(f"[WARN] {exc}")



if __name__ == "__main__":
    main()

