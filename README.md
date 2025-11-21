## Delta-K

Delta-K 将 GORS test-time scaling 论文中的推理、调度与训练逻辑统一收敛到 *单层目录*，方便快速查阅与集成。所有模块保持 PyTorch + HuggingFace Diffusers 实现，GPU 友好，中文注释到位。

### 模块划分（仅 6 个 Python 文件）

- `delta_k_utils.py`  
  注意力钩子 (`CrossAttentionCapture`)、Step 回调、概念/Tokenizer 映射、K-mean 统计以及 VLM present/missing 分析全部集中在此。
- `delta_k_core.py`  
  负责解析设备精度、构建 `DiffusionPipeline` 并封装 `run_diffusion_once`（包含 hook 注册与 latent 记录）。
- `delta_k_scheduler.py`  
  提供线性、VLM gap（triangle/plateau/burst）以及 mean-of-concept 调度，同时给出统一的 `get_schedule` 工厂函数。
- `delta_k_pipeline.py`  
  主流程：先跑 baseline 获取统计 → 调用 VLM 判断缺失概念 → 构建 delta-K 注入序列 → 再次生成最终图像。
- `delta_k_infer.py`  
  推理 CLI，内置默认配置 (`DEFAULT_CONFIG`)，支持通过 Python 配置文件或命令行覆盖所有参数，并内嵌 prompt 加载逻辑。
- `delta_k_training.py`  
  Prompt→Image 闭环脚本，输入 prompt 列表即可直接生成图片，无需外部数据集。

### 推理快速开始

```bash
python /root/Delta-K/delta_k_infer.py \
  --config_py /root/Delta-K/my_config.py \  # 可选，默认使用脚本内置配置
  --schedule mean_of_concept \
  --qwen_api_key sk-xxx \
  --prompt "a cozy living room with warm lights"
```

核心参数说明：
- `--config_py`：Python 文件需暴露 `CONFIG` 或 `DEFAULT_CONFIG` 字典；若为空则使用脚本默认配置。
- `--schedule`：`baseline`、`linear_large`、`vlm_gap_triangle|plateau|burst`、`mean_of_concept`。
- `--from_file` / `--prompt`：二选一，支持批量 txt 或单条 prompt。
- `--qwen_api_key`：用于 DashScope 兼容接口的 present/missing 分析，也可通过环境变量 `DASHSCOPE_API_KEY` 预设。
- 输出图片统一写入 `output.dir/samples`，文件名格式 `prompt_xxxxxx.png`。

### LoRA 闭环示例

```bash
python /root/Delta-K/delta_k_training.py \
  --model /root/autodl-tmp/SDXL1.0 \
  --prompt "neon cyberpunk street at night" \
  --schedule mean_of_concept \
  --qwen_api_key sk-xxx
```

- `--prompt` 与 `--from_file` 二选一，脚本会直接生成图片写入 `outdir/samples`。
- 不再依赖外部训练数据或 DataLoader，可结合自身 LoRA 权重进行闭环实验。

### 互相引用

所有源码位于同一目录，示例：

```python
from delta_k_pipeline import generate_image_with_schedule
from delta_k_scheduler import get_schedule
```

在外部工程中只需把 `Delta-K` 加入 `PYTHONPATH` 或复制所需文件，即可复用完整的 delta-K 流水线。

