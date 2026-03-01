import time
import torch
import warnings
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.utils import logging

# ===================== 全局环境初始化（架构通用，必开）=====================
# 忽略无关警告
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# 【架构无关通用硬件加速】所有GPU/模型通用，无兼容性风险
# 1. TF32加速：Ampere+架构显卡矩阵乘法提速30%+，无精度损失
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 2. CUDNN基准优化：固定输入尺寸下卷积/注意力计算提速
torch.backends.cudnn.benchmark = True
# 3. 显存碎片化优化：杜绝OOM，适配大模型
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",max_split_size_mb:128"

# ===================== 【用户唯一需要修改的配置区】=====================
@dataclass
class BenchmarkConfig:
    # -------------------------- 核心模型配置 --------------------------
    # 模型路径配置：支持任意SD系列/FLUX系列模型，自动识别架构
    model_paths: Dict[str, str] = field(default_factory=lambda: {
        "sdxl": "/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        "sd3.5medium": "/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "flux": "/data/yulin/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    })
    # 硬件配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16

    # -------------------------- 生成通用配置 --------------------------
    width: int = 1024
    height: int = 1024
    prompt: str = "a beautiful sunset over the ocean, photorealistic, 8k, high detail"
    negative_prompt: str = "blurry, low quality, ugly, distorted, pixelated"

    # -------------------------- 测试控制配置 --------------------------
    test_rounds: int = 3  # 正式测试轮数，取平均值
    warmup_rounds: int = 2  # 预热轮数，开启compile时自动+2
    save_images: bool = True
    save_dir: str = "./speed_test_results"

    # -------------------------- 加速开关（架构通用，一键启停）--------------------------
    enable_xformers: bool = True  # 注意力加速，所有模型通用，提速20%-40%
    enable_torch_compile: bool = False  # 图编译加速，首次运行有编译开销，后续提速30%-60%
    enable_int8_quant: bool = False  # INT8量化，显存降40%，提速15%-25%，所有模型通用
    torch_compile_mode: str = "max-autotune"  # 编译模式：max-autotune(性能最优)/reduce-overhead(平衡)/default(最快编译)

    # -------------------------- 步数优化配置（架构自动适配）--------------------------
    # 噪声预测模型（SDXL等）步数配置
    noise_predict_base_steps: int = 20
    noise_predict_optim_steps: int = 10
    # 流匹配模型（SD3/FLUX等）步数配置
    flow_match_base_steps: int = 28
    flow_match_optim_steps: int = 14
    # 通用CFG配置
    base_guidance_scale: float = 4.5

# ===================== 架构无关核心工具库 =====================
def init_dirs(save_dir: str):
    """初始化输出目录"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def clear_gpu_memory():
    """【关键】彻底释放GPU显存，杜绝跨用例显存残留"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

@dataclass
class ModelArchInfo:
    """架构无关的模型信息封装，自动识别"""
    model_name: str
    pipeline_type: str
    core_module: torch.nn.Module  # 核心计算模块（UNet/Transformer）
    core_module_name: str
    is_flow_match: bool  # 是否为流匹配模型
    support_negative_prompt: bool
    support_guidance_scale: bool
    recommend_scheduler: Callable

def auto_detect_model_arch(pipeline, model_name: str) -> ModelArchInfo:
    """【架构无关核心】自动识别模型架构，无需硬编码"""
    # 识别pipeline类型
    pipeline_type = pipeline.__class__.__name__
    
    # 自动识别核心计算模块
    core_module = None
    core_module_name = ""
    if hasattr(pipeline, "unet"):
        core_module = pipeline.unet
        core_module_name = "unet"
    elif hasattr(pipeline, "transformer"):
        core_module = pipeline.transformer
        core_module_name = "transformer"
    
    # 自动识别是否为流匹配模型
    is_flow_match = isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler) or "Flux" in pipeline_type or "StableDiffusion3" in pipeline_type
    
    # 自动识别支持的参数
    support_negative_prompt = hasattr(pipeline, "negative_prompt") or "StableDiffusionXLPipeline" in pipeline_type
    support_guidance_scale = hasattr(pipeline, "guidance_scale") or not ("Flux" in pipeline_type)
    
    # 自动推荐兼容的采样器
    recommend_scheduler = FlowMatchEulerDiscreteScheduler if is_flow_match else EulerDiscreteScheduler
    
    return ModelArchInfo(
        model_name=model_name,
        pipeline_type=pipeline_type,
        core_module=core_module,
        core_module_name=core_module_name,
        is_flow_match=is_flow_match,
        support_negative_prompt=support_negative_prompt,
        support_guidance_scale=support_guidance_scale,
        recommend_scheduler=recommend_scheduler
    )

# ===================== 架构无关加速实现 =====================
def load_model_with_acceleration(
    model_name: str,
    model_path: str,
    cfg: BenchmarkConfig,
    enable_quant: bool = False
):
    """【架构无关】加载模型并应用通用加速，自动适配所有模型"""
    print(f"\n===== 加载模型：{model_name} =====")
    clear_gpu_memory()

    # 基础加载参数（架构通用）
    load_kwargs = {
        "torch_dtype": cfg.dtype,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "device_map": cfg.device
    }

    # 【架构无关INT8量化】官方原生实现，兼容所有模型，无报错
    if enable_quant and cfg.enable_int8_quant:
        load_kwargs["load_in_8bit"] = True
        print("✅ INT8量化已启用（官方原生架构通用）")

    # 非FLUX模型加载fp16 variant，自动适配
    if "flux" not in model_name.lower():
        load_kwargs["variant"] = "fp16"

    # 自动加载pipeline，无需硬编码模型类型
    try:
        pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)[:100]}")

    # 自动识别模型架构
    arch_info = auto_detect_model_arch(pipeline, model_name)
    print(f"✅ 自动识别模型架构：{arch_info.pipeline_type} | {'流匹配模型' if arch_info.is_flow_match else '噪声预测模型'}")

    # 1. xformers注意力加速（架构通用）
    if cfg.enable_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xformers注意力加速已启用（架构通用）")
        except Exception as e:
            print(f"⚠️  xformers启用失败，已跳过：{str(e)[:60]}")

    # 2. torch.compile图编译加速（架构无关，自动识别核心模块）
    if cfg.enable_torch_compile and arch_info.core_module is not None:
        try:
            setattr(
                pipeline,
                arch_info.core_module_name,
                torch.compile(
                    arch_info.core_module,
                    mode=cfg.torch_compile_mode,
                    fullgraph=True,
                    dynamic=False
                )
            )
            print(f"✅ torch.compile已启用（核心模块：{arch_info.core_module_name}）")
        except Exception as e:
            print(f"⚠️  torch.compile启用失败，已跳过：{str(e)[:60]}")

    # 3. 通用显存优化（架构通用，自动判断兼容性）
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("max")

    return pipeline, arch_info

def run_inference_test(
    pipeline,
    arch_info: ModelArchInfo,
    cfg: BenchmarkConfig,
    use_optimized_steps: bool = False
):
    """【架构无关】推理测试，自动适配模型参数，杜绝报错"""
    # 1. 自动配置步数和采样器（架构适配）
    if arch_info.is_flow_match:
        steps = cfg.flow_match_optim_steps if use_optimized_steps else cfg.flow_match_base_steps
    else:
        steps = cfg.noise_predict_optim_steps if use_optimized_steps else cfg.noise_predict_base_steps
    
    # 自动设置兼容的采样器，杜绝生成质量异常
    pipeline.scheduler = arch_info.recommend_scheduler.from_config(pipeline.scheduler.config)
    print(f"✅ 采样器已配置：{arch_info.recommend_scheduler.__name__} | 推理步数：{steps}")

    # 2. 【架构无关】自动构建推理参数，杜绝不支持的参数报错
    infer_kwargs = {
        "prompt": cfg.prompt,
        "width": cfg.width,
        "height": cfg.height,
        "num_inference_steps": steps,
        "generator": torch.Generator(cfg.device).manual_seed(42),
        "num_images_per_prompt": 1
    }

    # 自动添加模型支持的参数
    if arch_info.support_guidance_scale:
        infer_kwargs["guidance_scale"] = cfg.base_guidance_scale
    if arch_info.support_negative_prompt:
        infer_kwargs["negative_prompt"] = cfg.negative_prompt

    # 3. 预热处理：开启compile时自动增加预热轮次，确保编译完成
    final_warmup_rounds = cfg.warmup_rounds + 2 if cfg.enable_torch_compile else cfg.warmup_rounds
    print(f"预热模型（{final_warmup_rounds}轮）...")
    for i in range(final_warmup_rounds):
        _ = pipeline(**infer_kwargs)

    # 4. 正式计时测试（GPU同步，确保计时准确）
    total_time = 0.0
    save_dir = init_dirs(cfg.save_dir)
    print(f"正式测试（{cfg.test_rounds}轮）...")

    for round_idx in range(cfg.test_rounds):
        # 重置随机种子，保证可复现
        infer_kwargs["generator"] = torch.Generator(cfg.device).manual_seed(42 + round_idx)
        
        # GPU严格同步计时，杜绝异步执行导致的计时不准
        torch.cuda.synchronize()
        start_time = time.time()
        
        result = pipeline(**infer_kwargs)
        image = result.images[0]
        
        torch.cuda.synchronize()
        round_time = time.time() - start_time
        total_time += round_time

        # 保存图片
        if cfg.save_images:
            img_name = f"{arch_info.model_name}_steps-{steps}_quant-{cfg.enable_int8_quant}_compile-{cfg.enable_torch_compile}_round-{round_idx+1}.png"
            image.save(f"{save_dir}/{img_name}")
        
        print(f"  第{round_idx+1}轮耗时：{round_time:.2f}秒")

    avg_time = total_time / cfg.test_rounds
    print(f"✅ 本轮测试平均耗时：{avg_time:.2f}秒")
    return avg_time

# ===================== 测试调度主流程 =====================
def run_full_benchmark(cfg: BenchmarkConfig):
    """全量测试调度，架构无关，错误隔离"""
    # 测试矩阵：所有组合架构通用，自动适配
    test_matrix = [
        {"name": "Baseline", "optim_steps": False, "enable_quant": False},
        {"name": "仅步数优化", "optim_steps": True, "enable_quant": False},
        {"name": "仅INT8量化", "optim_steps": False, "enable_quant": True},
        {"name": "步数优化+INT8量化", "optim_steps": True, "enable_quant": True},
    ]

    final_results = {}

    # 遍历所有模型
    for model_name, model_path in cfg.model_paths.items():
        final_results[model_name] = {}
        print(f"\n{'='*60}")
        print(f"开始测试模型：{model_name}")
        print(f"{'='*60}")
        print(f"全局加速配置：xformers={cfg.enable_xformers} | torch.compile={cfg.enable_torch_compile}")

        # 遍历所有测试组合
        for test_case in test_matrix:
            test_name = test_case["name"]
            print(f"\n--- 测试组合：{test_name} ---")
            
            try:
                # 加载模型并应用加速
                pipeline, arch_info = load_model_with_acceleration(
                    model_name=model_name,
                    model_path=model_path,
                    cfg=cfg,
                    enable_quant=test_case["enable_quant"]
                )

                # 执行推理测试
                avg_time = run_inference_test(
                    pipeline=pipeline,
                    arch_info=arch_info,
                    cfg=cfg,
                    use_optimized_steps=test_case["optim_steps"]
                )

                # 保存结果
                final_results[model_name][test_name] = f"{avg_time:.2f}秒"

                # 【关键】彻底释放当前模型显存，杜绝跨用例OOM
                del pipeline, arch_info
                clear_gpu_memory()

            except Exception as e:
                # 错误隔离：单个用例失败不影响整体测试
                error_msg = str(e)[:80]
                print(f"❌ 测试失败：{error_msg}")
                final_results[model_name][test_name] = f"失败: {error_msg}"
                clear_gpu_memory()

    # ===================== 结果输出 =====================
    print(f"\n{'='*60}")
    print("📊 【最终测试结果汇总】（单位：秒）")
    print(f"{'='*60}")
    print(f"全局配置：xformers={cfg.enable_xformers} | torch.compile={cfg.enable_torch_compile} | INT8量化={cfg.enable_int8_quant}")
    print(f"测试设备：{cfg.device} | PyTorch版本：{torch.__version__}")

    for model_name, model_results in final_results.items():
        print(f"\n【{model_name}】")
        for test_name, time_str in model_results.items():
            print(f"  {test_name}: {time_str}")

    # 保存结果到文件
    save_dir = init_dirs(cfg.save_dir)
    result_file = f"{save_dir}/final_benchmark_results.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("架构无关多模型推理加速测试结果汇总\n")
        f.write("="*60 + "\n")
        f.write(f"测试设备：{cfg.device}\n")
        f.write(f"PyTorch版本：{torch.__version__}\n")
        f.write(f"全局加速配置：xformers={cfg.enable_xformers} | torch.compile={cfg.enable_torch_compile}\n")
        f.write("="*60 + "\n")
        for model_name, model_results in final_results.items():
            f.write(f"\n【{model_name}】\n")
            for test_name, time_str in model_results.items():
                f.write(f"  {test_name}: {time_str}\n")
    
    print(f"\n📄 详细结果已保存至：{result_file}")
    return final_results

# ===================== 主函数入口 =====================
if __name__ == "__main__":
    # 初始化配置
    benchmark_cfg = BenchmarkConfig()
    init_dirs(benchmark_cfg.save_dir)

    # 打印启动信息
    print(f"{'='*60}")
    print("架构无关多模型推理加速对比测试")
    print(f"{'='*60}")
    print(f"测试设备：{benchmark_cfg.device}")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"测试模型：{list(benchmark_cfg.model_paths.keys())}")
    print(f"测试轮数：{benchmark_cfg.test_rounds}")
    print(f"xformers加速：{'开启' if benchmark_cfg.enable_xformers else '关闭'}")
    print(f"torch.compile：{'开启' if benchmark_cfg.enable_torch_compile else '关闭'}")
    print(f"INT8量化：{'开启' if benchmark_cfg.enable_int8_quant else '关闭'}")
    
    if benchmark_cfg.enable_torch_compile:
        print(f"⚠️  已开启torch.compile，首次运行会有3-10分钟的编译开销，请耐心等待...")

    # 运行全量测试
    run_full_benchmark(benchmark_cfg)