import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

def load_filtered_data(file_path):
    rows = []
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                meta = data.get('metadata', {})
                stats_data = data.get('stats', {})
                
                # --- 核心过滤逻辑 ---
                # 1. 检查 metadata 中是否存在 missing 和 present 的概念
                missing_concepts = meta.get('missing', [])
                present_concepts = meta.get('present', [])
                
                # 如果其中一个为空，说明没有对比基准，跳过此条数据
                if not missing_concepts or not present_concepts:
                    continue
                
                # 2. 检查统计数值是否有效 (防止 mean 为 0 的异常数据)
                p_stats = stats_data.get('present', {})
                m_stats = stats_data.get('missing', {})
                
                if not p_stats or not m_stats or p_stats.get('present_mean', 0) == 0:
                    continue

                # 提取数据并归一化列名
                prompt = meta.get('prompt', 'Unknown')
                
                # 处理 Present
                p_row = {k.replace('present_', ''): v for k, v in p_stats.items()}
                p_row.update({'group': 'Present', 'prompt': prompt})
                rows.append(p_row)
                
                # 处理 Missing
                m_row = {k.replace('missing_', ''): v for k, v in m_stats.items()}
                m_row.update({'group': 'Missing', 'prompt': prompt})
                rows.append(m_row)
                
            except Exception as e:
                print(f"解析行失败: {e}")
                continue
    return pd.DataFrame(rows)

def run_statistical_report(df):
    if df.empty:
        print("没有可供分析的有效数据（可能被过滤掉了）。")
        return

    metrics = ['mean', 'max', 'p90', 'sparsity', 'cv']
    
    # 计算配对差异（按 Prompt 分组计算 M/P 比例）
    # 这样可以看出在同一个 Prompt 下，Missing 到底比 Present 弱多少
    print(f"\n{'指标':<15} | {'Present 均值':<12} | {'Missing 均值':<12} | {'P-Value':<10} | {'结论'}")
    print("-" * 75)
    print(len(df))
    for m in metrics:
        p_vals = df[df['group'] == 'Present'][m]
        m_vals = df[df['group'] == 'Missing'][m]
        
        # 使用相关样本 T 检验 (Paired T-Test)，因为它们来自同一个 Prompt
        t_stat, p_val = stats.ttest_rel(p_vals, m_vals) if len(p_vals) == len(m_vals) else stats.ttest_ind(p_vals, m_vals)
        
        sig = "⭐ 显著差异" if p_val < 0.05 else "不显著"
        print(f"{m:<17} | {p_vals.mean():.6f} | {m_vals.mean():.6f} | {p_val:.4f} | {sig}")

def plot_distributions(df):
    metrics = ['mean', 'max', 'p90', 'cv']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sns.set_palette("husl")

    for i, m in enumerate(metrics):
        sns.violinplot(x='group', y=m, data=df, ax=axes[i], inner="quart")
        axes[i].set_title(f'Distribution of {m.upper()}')
    
    plt.tight_layout()
    plt.savefig('filtered_attention_analysis.png')
    plt.show()

if __name__ == "__main__":
    df = load_filtered_data("/data/yulin/DK/Delta-K/logs/attention_data.jsonl")
    run_statistical_report(df)
    plot_distributions(df)