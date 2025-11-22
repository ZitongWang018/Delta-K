## Delta-K

We propose Delta-K, a training-free, plug-and-play framework that enhances compositional text-image alignment in Diffusion Transformers (DiTs). By analyzing baseline generations with a Vision-Language Model (VLM) to identify missing concepts, Delta-K computes differential key vectors (∆K) from masked prompts and injects them into cross-attention layers with a dynamically optimized schedule. This approach directly manipulates the key space to amplify under-represented concepts, significantly improving the generation of complex multi-instance prompts without additional training or architectural changes. This repository is the official implementation of Delta-K.

### Quick Start

```bash
python delta_k_infer.py \
  --config_py my_config.py \
  --schedule mean_of_concept \
  --qwen_api_key YOUR_API_KEY \
  --prompt "a cozy living room with warm lights"
```
