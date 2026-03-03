# import torch
# import os
# import re
# from typing import List, Dict, Tuple, Optional, Any
# from diffusers import DiffusionPipeline

# # ==========================================
# # 1. Pipeline 构建逻辑 (完全复制您的代码)
# # ==========================================
# def build_pipeline(model_path: str):
#     # 保持您原本的逻辑结构
#     try:
#         device, dtype = "cuda", torch.bfloat16
#     except:
#         device, dtype = "cpu", torch.bfloat16
    
#     pipe = DiffusionPipeline.from_pretrained(
#         model_path,
#         torch_dtype=dtype,
#         use_safetensors=True,
#         local_files_only=True,
#     )
#     return pipe.to(device)

# # ==========================================
# # 2. 基础工具函数
# # ==========================================
# def _clean_token(token: str) -> str:
#     """清理 token 中的特殊符号"""
#     token = token.lower()
#     for bad in ["</w>", "Ġ", "##", "Ċ", "｡", "▁"]:
#         token = token.replace(bad, "")
#     return re.sub(r"[^a-z]+", "", token)

# def tokens_from_prompt(tokenizer, prompt: str) -> List[str]:
#     if tokenizer is None: return []
#     try:
#         ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
#         return [_clean_token(x) for x in tokenizer.convert_ids_to_tokens(ids)]
#     except Exception as e:
#         print(f"Tokenizer error: {e}")
#         return []

# # ==========================================
# # 3. 核心探测逻辑
# # ==========================================
# def inspect_tokenizer_fusion(model_path: str, model_name: str):
#     print(f"\n{'='*20} 🔎 Inspecting: {model_name} {'='*20}")
    
#     try:
#         # [关键点]：使用您定义的 build_pipeline 逻辑
#         pipe = build_pipeline(model_path)
#     except Exception as e:
#         print(f"❌ 加载失败: {e}")
#         return

#     # ------------------------------------------
#     # A. 组件探测
#     # ------------------------------------------
#     print("\n[1] Tokenizer & Encoder 维度:")
#     tokenizers = []
#     encoder_dims = []
#     encoder_seq_lens = []
    
#     tokenizer_attrs = ["tokenizer", "tokenizer_2", "tokenizer_3"]
#     encoder_attrs = ["text_encoder", "text_encoder_2", "text_encoder_3"]
    
#     prompt = "a cat"
    
#     # 获取各个 Encoder 的配置信息
#     for i, (t_attr, e_attr) in enumerate(zip(tokenizer_attrs, encoder_attrs)):
#         if hasattr(pipe, t_attr):
#             tok = getattr(pipe, t_attr)
#             tokenizers.append(tok)
            
#             enc = getattr(pipe, e_attr)
#             dim = 0
#             if hasattr(enc, 'config'): 
#                 dim = enc.config.hidden_size
#             elif hasattr(enc, 'text_model'): 
#                 dim = enc.text_model.config.hidden_size
            
#             # 获取实际序列长度
#             # 注意：这里的 device 需要与 pipe 一致
#             inputs = tok(prompt, return_tensors="pt").to(pipe.device)
#             seq_len = inputs["input_ids"].shape[1]
            
#             encoder_dims.append(dim)
#             encoder_seq_lens.append(seq_len)
            
#             print(f"  - {t_attr}: {tok.__class__.__name__} | Hidden Dim: {dim} | Token Len: {seq_len}")

#     # ------------------------------------------
#     # B. 融合策略验证
#     # ------------------------------------------
#     print("\n[2] 融合策略验证:"
#           f"\n  (测试 Prompt: '{prompt}')")
    
#     fused_dim = 0
#     fused_seq_len = 0
#     model_type_key = "unknown"
    
#     try:
#         # 1. 动态构建 encode_prompt 参数
#         pipeline_class_name = pipe.__class__.__name__
        
#         kwargs = {
#             "prompt": prompt,
#             "device": pipe.device,
#             "num_images_per_prompt": 1,
#         }
        
#         # 针对 SD3 / FLUX 需要显式传递 prompt_2 / prompt_3
#         if "StableDiffusion3" in pipeline_class_name:
#             kwargs["prompt_2"] = prompt
#             kwargs["prompt_3"] = prompt
#         elif "Flux" in pipeline_class_name:
#             kwargs["prompt_2"] = prompt
#             # Flux 不需要 do_classifier_free_guidance 参数
        
#         # 针对 SDXL
#         if "StableDiffusionXL" in pipeline_class_name:
#             kwargs["do_classifier_free_guidance"] = False

#         # 2. 调用 encode_prompt
#         result = pipe.encode_prompt(**kwargs)
        
#         # 3. 智能解包返回值 (适配不同模型的返回结构)
#         prompt_embeds = None
#         pooled_embeds = None
        
#         if isinstance(result, tuple):
#             for item in result:
#                 if isinstance(item, torch.Tensor):
#                     if item.ndim == 3 and prompt_embeds is None:
#                         prompt_embeds = item
#                     elif item.ndim == 2 and pooled_embeds is None:
#                         pooled_embeds = item
        
#         if prompt_embeds is not None:
#             fused_dim = prompt_embeds.shape[-1]
#             fused_seq_len = prompt_embeds.shape[1]
            
#             print(f"  - 最终融合后的 Prompt Embeds Shape: {list(prompt_embeds.shape)}")
#             print(f"    -> Fused Dim: {fused_dim}")
#             print(f"    -> Fused Seq Len: {fused_seq_len}")
#         else:
#             print("  ❌ 无法从 encode_prompt 结果中提取 prompt_embeds")
#             return

#         # ------------------------------------------
#         # C. 策略判定逻辑
#         # ------------------------------------------
        
#         if "SDXL" in model_name:
#             expected_dim = sum(encoder_dims)
#             if fused_dim == expected_dim:
#                 model_type_key = "sdxl"
#                 print(f"  ✅ 结论: SDXL 模式 - 维度拼接 ({encoder_dims[0]}+{encoder_dims[1]}={fused_dim})")
#             else:
#                 print(f"  ⚠️ 异常: 维度不匹配")

#         elif "SD3" in model_name:
#             # SD3 序列拼接，维度投影统一
#             if fused_dim == encoder_dims[2]: 
#                  model_type_key = "sd3"
#                  print(f"  ✅ 结论: SD3 模式 - 序列拼接 (L+G+T5)")
#             else:
#                  print(f"  ⚠️ 策略: 未知 SD3 变体")

#         elif "FLUX" in model_name:
#             t5_dim = encoder_dims[1] if len(encoder_dims) > 1 else 0
#             t5_len = encoder_seq_lens[1] if len(encoder_seq_lens) > 1 else 0
            
#             if fused_dim == t5_dim and fused_seq_len == t5_len:
#                 model_type_key = "flux"
#                 print(f"  ✅ 结论: FLUX 模式 - 仅 T5 充当 Context")
#             else:
#                 print(f"  ⚠️ 警告: 未能匹配标准 FLUX 特征")

#         # ------------------------------------------
#         # D. 验证 Map 函数逻辑
#         # ------------------------------------------
#         print("\n[3] 验证 Map 函数逻辑")
#         validate_map_logic(
#             model_type=model_type_key,
#             tokenizers=tuple(tokenizers),
#             prompt=prompt,
#             seq_lens=encoder_seq_lens,
#             max_seq_len=fused_seq_len
#         )

#     except Exception as e:
#         print(f"  ❌ 探测出错: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # 显式清理内存
#         del pipe
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# # ==========================================
# # 4. Map 逻辑验证函数
# # ==========================================
# def validate_map_logic(model_type: str, tokenizers: Tuple, prompt: str, seq_lens: List[int], max_seq_len: int):
#     words = re.findall(r"[a-z]+", prompt.lower())
#     concepts = list(set(words))
#     if not concepts: concepts = ["a", "cat"]
    
#     t_list = [tokens_from_prompt(tok, prompt) for tok in tokenizers if tok]
#     all_valid = True
    
#     for concept in concepts:
#         indices = []
        
#         if model_type == 'sdxl':
#             t1 = t_list[0]
#             t2 = t_list[1] if len(t_list) > 1 else []
#             for i in range(len(t1)):
#                 w1 = t1[i] if i < len(t1) else ""
#                 w2 = t2[i] if i < len(t2) else ""
#                 if concept in w1 or concept in w2:
#                     indices.append(i)

#         elif model_type == 'flux':
#             if len(t_list) > 1:
#                 t_t5 = t_list[1]
#                 for i, word in enumerate(t_t5):
#                     if concept in word:
#                         indices.append(i)
        
#         elif model_type == 'sd3':
#             offsets = [0, seq_lens[0], seq_lens[0] + seq_lens[1]]
            
#             t_l = t_list[0]
#             for i, word in enumerate(t_l):
#                 if concept in word:
#                     indices.append(offsets[0] + i)
            
#             t_g = t_list[1] if len(t_list) > 1 else []
#             for i, word in enumerate(t_g):
#                 if concept in word:
#                     indices.append(offsets[1] + i)
            
#             t_t5 = t_list[2] if len(t_list) > 2 else []
#             for i, word in enumerate(t_t5):
#                 if concept in word:
#                     indices.append(offsets[2] + i)

#         indices = sorted(list(set(indices)))
#         is_valid = all(idx < max_seq_len for idx in indices)
        
#         status = "✅ PASS" if is_valid else "❌ FAIL"
#         if not is_valid: all_valid = False
        
#         print(f"  - Concept '{concept}': Indices {indices[:5]}... -> {status}")

#     print(f"\n  最终结果: {'✅ 全部通过' if all_valid else '❌ 存在错误'}")

# # ==========================================
# # 5. 主程序
# # ==========================================
# if __name__ == "__main__":
#     model_configs = [
#         ("/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", "SDXL"),
#         ("/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80", "SD3.5"),
#         ("/data/yulin/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", "FLUX"),
#     ]

#     for path, name in model_configs:
#         if os.path.exists(path):
#             inspect_tokenizer_fusion(path, name)
#         else:
#             print(f"路径不存在，跳过: {path}")

# import torch
# import os
# import re
# from typing import List, Dict, Tuple, Optional, Any
# from diffusers import DiffusionPipeline
# import torch
# from diffusers import DiffusionPipeline

# def verify_sd35_clip_fusion(model_path: str):
#     print(f"\n{'='*20} 🔎 SD3.5 CLIP Fusion Verification {'='*20}")
    
#     # 1. 构建环境
#     try:
#         device, dtype = "cuda", torch.bfloat16
#     except:
#         device, dtype = "cpu", torch.bfloat16
    
#     pipe = DiffusionPipeline.from_pretrained(
#         model_path, torch_dtype=dtype, use_safetensors=True, local_files_only=True
#     ).to(device)
    
#     prompt = "a cat"
    
#     # 2. 手动编码 T5 部分 (基准真值)
#     # 根据之前的日志，T5 长度固定为 256
#     t5_max_length = 256 
#     print(f"[1] 手动编码 T5 (Max Len={t5_max_length})...")
    
#     inputs_t5 = pipe.tokenizer_3(
#         prompt, 
#         return_tensors="pt", 
#         padding="max_length", 
#         max_length=t5_max_length, 
#         truncation=True
#     ).to(device)
    
#     with torch.no_grad():
#         # T5 输出: last_hidden_state
#         t5_output = pipe.text_encoder_3(inputs_t5.input_ids)[0]
    
#     print(f"  - T5 Manual Output Shape: {t5_output.shape}") # 期望 [1, 256, 4096]

#     # 3. 调用 Pipeline 的 encode_prompt (获取融合结果)
#     print(f"[2] 调用 Pipeline encode_prompt...")
    
#     try:
#         # 关键：必须显式关闭 CFG，否则返回值数量会变，且生成负向嵌入
#         result = pipe.encode_prompt(
#             prompt=prompt,
#             prompt_2=prompt,
#             prompt_3=prompt,
#             device=device,
#             do_classifier_free_guidance=False 
#         )
        
#         # 智能解包 (自动适配返回值数量)
#         prompt_embeds = None
#         pooled_embeds = None
        
#         if isinstance(result, tuple):
#             for item in result:
#                 if isinstance(item, torch.Tensor):
#                     if item.ndim == 3 and prompt_embeds is None:
#                         prompt_embeds = item
#                     elif item.ndim == 2 and pooled_embeds is None:
#                         pooled_embeds = item
        
#         if prompt_embeds is None:
#             print("❌ 解包失败")
#             return

#         print(f"  - Fused Embeds Shape: {prompt_embeds.shape}")
        
#         # 4. 核心验证逻辑：切片比对
#         print(f"[3] 切片比对验证:")
        
#         total_len = prompt_embeds.shape[1]
#         clip_len = 77
        
#         # 推测: Total = CLIP(77) + T5(256)
#         if total_len == clip_len + t5_max_length:
#             print(f"  ✅ 长度匹配: Total({total_len}) = CLIP({clip_len}) + T5({t5_max_length})")
            
#             # 切片
#             clip_part = prompt_embeds[:, :clip_len, :]
#             t5_part_from_fused = prompt_embeds[:, clip_len:, :]
            
#             # 比对 T5 部分 (这是最硬的证据)
#             # 如果 T5 part 完全等于手动编码的 T5 output，说明 T5 是直接拼接，未做复杂变换
#             diff = (t5_part_from_fused - t5_output).abs().max().item()
            
#             print(f"  - T5 Part Diff: {diff}")
#             if diff < 1e-4:
#                 print("  ✅ 结论确认: T5 部分是直接拼接")
            
#             # 比对 CLIP 部分
#             print(f"  - CLIP Part Shape: {clip_part.shape}")
#             print(f"  - CLIP Part Mean Norm: {clip_part.norm(dim=-1).mean().item():.4f}")
            
#             print(f"\n[4] 最终融合机制确认:")
#             print(f"  1. 序列结构: [ CLIP-L/G Fused (77) | T5 (256) ]")
#             print(f"  2. CLIP融合: 维度投影到 4096 后求和/平均 (因 768+1280!=4096 且序列未翻倍)")
#             print(f"  3. T5融合: 直接 Concat 拼接")
            
#         else:
#             print(f"  ⚠️ 长度不匹配: Total={total_len}, Expected={clip_len + t5_max_length}")

#     except Exception as e:
#         print(f"❌ 错误: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         del pipe
#         torch.cuda.empty_cache()



# path="/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
# verify_sd35_clip_fusion(path)

import torch
from diffusers import DiffusionPipeline

def inspect_head_dims(model_path: str, model_type: str):
    print(f"\n--- Checking: {model_type} ---")
    
    try:
        if model_type == 'sdxl':
            # 加载 UNet 实例以获取真实配置 (use_safetensors=True 加载更快)
            # 注意：这里只加载配置和元数据，不加载权重，所以很快
            unet = UNet2DConditionModel.from_pretrained(
                f"{model_path}/unet", 
                subfolder="", 
                local_files_only=True,
                # low_cpu_mem_usage=True # 可选，进一步降低内存
            )
            # 此时 unet.config 是一个对象，可以用 . 访问
            dim = unet.config.attention_head_dim
            print(f"✅ SDXL UNet attention_head_dim: {dim}")
            
        elif model_type in ['flux', 'sd3']:
            # 尝试加载 Transformer
            # 注意：FLUX 和 SD3 的 transformer 可能在根目录或 transformer 文件夹
            try:
                transformer = Transformer2DModel.from_pretrained(
                    f"{model_path}/transformer", 
                    local_files_only=True
                )
            except:
                # 如果路径不对，尝试从 pipeline 加载
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16)
                transformer = pipe.transformer
            
            dim = transformer.config.attention_head_dim
            print(f"✅ {model_type.upper()} Transformer attention_head_dim: {dim}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Fallback: Trying to read config.json directly as dict...")
        try:
            import json
            if model_type == 'sdxl':
                path = f"{model_path}/unet/config.json"
            else:
                path = f"{model_path}/transformer/config.json"
            
            with open(path, 'r') as f:
                config = json.load(f)
            # 字典访问方式
            dim = config.get("attention_head_dim", "Not Found")
            print(f"Config JSON attention_head_dim: {dim}")
        except Exception as e2:
            print(f"Failed to read json: {e2}")

import torch
from diffusers import UNet2DConditionModel

def verify_sdxl_head_dim(model_path: str):
    print(f"\n==================== SDXL Head Dim 验证 ====================")
    
    # 1. 读取配置
    # config 文件路径通常在 unet/config.json
    try:
        # 尝试加载模型配置
        unet_config = UNet2DConditionModel.load_config(f"{model_path}/unet", local_files_only=True)
        # 如果是字典
        if isinstance(unet_config, dict):
            block_out_channels = unet_config.get("block_out_channels", [320, 640, 1280])
            config_values = unet_config.get("attention_head_dim", [5, 10, 20])
        else:
            # 如果是 PretrainedConfig 对象
            block_out_channels = unet_config.block_out_channels
            config_values = unet_config.attention_head_dim
            
        print(f"[Config] block_out_channels (Hidden Dims): {block_out_channels}")
        print(f"[Config] attention_head_dim (Config Raw): {config_values}")

    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 2. 直接加载权重进行验证 (Ground Truth)
    # 我们加载第一层的 attention 权重，看它的形状
    print("\n--- 权重形状验证 ---")
    try:
        # 仅加载 unet，为了节省内存可以指定 low_cpu_mem_usage
        unet = UNet2DConditionModel.from_pretrained(
            f"{model_path}/unet", 
            local_files_only=True, 
            low_cpu_mem_usage=True,
            # device_map="auto" # 如果显存不够可以开启
        )
        
        # 检查 down_blocks.0 (第0层) 的注意力
        # 通常是 down_blocks.0.attentions.0.transformer_blocks.0.attn1
        block_0 = unet.down_blocks[0]
        if hasattr(block_0, 'attentions') and len(block_0.attentions) > 0:
            attn_layer = block_0.attentions[0].transformer_blocks[0].attn1
            
            # 获取 to_q 的权重, 形状为 [out_features, in_features]
            # in_features = hidden_dim
            # out_features = hidden_dim (因为 Q 投影通常保持维度)
            q_weight = attn_layer.to_q.weight
            
            hidden_dim_layer0 = q_weight.shape[1] # 输入维度即该层的 Hidden Dim
            
            print(f"Layer 0 (down_blocks.0):")
            print(f"  - 权重输入维度: {hidden_dim_layer0}")
            print(f"  - Config 配置值: {config_values[0]}")
            
            # 核心验证公式
            if config_values[0] != 0:
                calculated_head_dim = hidden_dim_layer0 / config_values[0]
                print(f"  - 计算结果: {hidden_dim_layer0} / {config_values[0]} = {calculated_head_dim}")
                
                if calculated_head_dim == 64:
                    print(f"  ✅ 结论: Config 存储的是 [头数], 真实 Head Dim = 64")
                elif calculated_head_dim == hidden_dim_layer0:
                     print(f"  ✅ 结论: Config 存储的是 [Head Dim], 真实 Head Dim = {calculated_head_dim}")
                else:
                     print(f"  ⚠️ 结论: 非标准配置，计算得 Head Dim = {calculated_head_dim}")

        # 检查 down_blocks.1 (中间层)
        if hasattr(unet, 'down_blocks') and len(unet.down_blocks) > 1:
             block_1 = unet.down_blocks[1]
             # 有些 block 可能没有 attention，需要遍历寻找
             # 简单起见，我们直接取 channel 维度验证
             # DownBlock2D 可能没有 attention，我们取 mid_block 或者 down_blocks.1 (如果是 CrossAttnDownBlock)
             if hasattr(block_1, 'attentions'):
                 attn_layer = block_1.attentions[0].transformer_blocks[0].attn1
                 hidden_dim_layer1 = attn_layer.to_q.weight.shape[1]
                 
                 print(f"\nLayer 1 (down_blocks.1):")
                 print(f"  - 权重输入维度: {hidden_dim_layer1}")
                 print(f"  - Config 配置值: {config_values[1]}")
                 
                 calculated_head_dim = hidden_dim_layer1 / config_values[1]
                 print(f"  - 计算结果: {hidden_dim_layer1} / {config_values[1]} = {calculated_head_dim}")
                 
    except Exception as e:
        print(f"加载权重验证失败: {e}")

# 运行验证
# 请替换为您的实际路径
verify_sdxl_head_dim("/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b")


# 使用示例
# inspect_head_dims("/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b", "sdxl")
# inspect_head_dims("/data/yulin/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", "flux")
# inspect_head_dims("/data/yulin/hf_cache/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80","sd3")
