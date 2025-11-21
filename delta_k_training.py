import argparse
from pathlib import Path
from typing import List, Optional

from delta_k_pipeline import generate_image_with_schedule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/SDXL1.0")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="/root/autodl-tmp/DeltaK_LoRA")
    parser.add_argument("--schedule", type=str, default="mean_of_concept")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--qwen_api_key", type=str, default=None)
    return parser.parse_args()


def _load_prompts(prompt: Optional[str], prompt_file: Optional[str], batch_size: int) -> List[List[str]]:
    if prompt_file:
        file_path = Path(prompt_file)
        if not file_path.exists():
            raise FileNotFoundError(f"找不到 prompt 文件: {file_path}")
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return [[line.strip().split("\t")[0]] * batch_size for line in lines]
    if not prompt:
        raise ValueError("必须提供 prompt 或 prompt 文件")
    return [[prompt] * batch_size]


def main():
    args = parse_args()
    prompts = _load_prompts(args.prompt, args.from_file, args.batch_size)
    outdir = Path(args.outdir)
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    counter = len(list(sample_dir.glob("*.png")))
    errors = 0
    for batch_prompts in prompts:
        for iteration in range(args.n_iter):
            try:
                image = generate_image_with_schedule(
                    model_path=args.model,
                    prompt=batch_prompts[0],
                    schedule=args.schedule,
                    steps=args.steps,
                    seed=args.seed + iteration,
                    qwen_api_key=args.qwen_api_key,
                    schedule_config={},
                )
                filename = f"{batch_prompts[0]}_{counter:06d}.png"
                image.save(sample_dir / filename)
                counter += 1
            except Exception as exc:
                errors += 1
                print(f"[WARN] 生成失败：{exc}")
    print(f"生成完成，失败次数：{errors}")


if __name__ == "__main__":
    main()

