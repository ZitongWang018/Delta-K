import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os

def load_comprehensive_data(file_path):
    overall_rows = []
    step_rows = []
    
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return pd.DataFrame(), pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                meta = data.get('metadata', {})
                overall = data.get('overall_stats', {})
                temporal = data.get('temporal_flow', {})
                m_concept=meta.get('missing',[])
                p_concept=meta.get('present',[])
                is_m=True
                is_p=True
                for word in m_concept:
                    if len(word.split(' '))>3:
                        print(word)
                        is_m=False
                for word in p_concept:
                    if len(word.split(' '))>3:
                        is_p=False
                # --- 核心过滤逻辑 ---
                p_overall = overall.get('present', {})
                m_overall = overall.get('missing', {})
                
                # 过滤掉缺失项全为 0 的无效数据（索引未匹配）
                prompt = meta.get('prompt', 'Unknown')
                if m_overall.get('missing_mean', 0) != 0 and is_m:
                    row = {k.split('_')[-1]: v for k, v in m_overall.items()}
                    row.update({'group': 'Missing', 'prompt': prompt})
                    overall_rows.append(row)
                if p_overall.get('present_mean', 0) != 0 and is_p:
                    row = {k.split('_')[-1]: v for k, v in p_overall.items()}
                    row.update({'group': 'Present', 'prompt': prompt})
                    overall_rows.append(row)
                

                # # 1. 处理全局统计 (Overall)
                # for group, stats_dict in [('Present', p_overall), ('Missing', m_overall)]:
                    

                # 2. 处理时序统计 (Step-wise)
                p_steps = temporal.get('present_steps', {})
                m_steps = temporal.get('missing_steps', {})
                tmp=0
                for sid in p_steps.keys():
                    # 记录 Present Step
                    if p_steps[sid]['mean']!=0.0 and is_p:
                        step_rows.append({
                            'step': int(sid),
                            'group': 'Present',
                            'mean': p_steps[sid]['mean'],
                            'max': p_steps[sid]['max'],
                            'var': p_steps[sid].get('var', 0),
                            'prompt': prompt
                        })
                    tmp+=p_steps[sid]['mean']
                    # 记录 Missing Step (如果存在)
                for sid in m_steps.keys():
                    if m_steps[sid]['mean']!=0.0 and is_m:
                        step_rows.append({
                            'step': int(sid),
                            'group': 'Missing',
                            'mean': m_steps[sid]['mean'],
                            'max': m_steps[sid]['max'],
                            'var': m_steps[sid].get('var', 0),
                            'prompt': prompt
                        })

            except Exception as e:
                print(f"解析行失败: {e}")
                continue
    df_overall = pd.DataFrame(overall_rows)
    df_steps = pd.DataFrame(step_rows)
    if not df_overall.empty:
        print(f"成功加载有效样本: {len(df_overall)}")
    return df_overall, df_steps

def run_statistical_report(df):
    if df.empty:
        print("无有效数据。")
        return

    # 包含方差 var 进行分析
    metrics = ['mean', 'max', 'p90', 'var', 'cv', 'sparsity']
    print(f"\n{'指标':<15} | {'Present 均值':<12} | {'Missing 均值':<12} | {'P-Value':<10} | {'结论'}")
    print("-" * 85)

    for m in metrics:
        if m not in df.columns:
            continue
        
        # 提取两组数据并剔除缺失值
        p_vals = df[df['group'] == 'Present'][m].dropna()
        m_vals = df[df['group'] == 'Missing'][m].dropna()
        
        p_mean = p_vals.mean()
        m_mean = m_vals.mean()
        p_val = float('nan')  # 初始化p值为N/A
    
        if len(p_vals) < 3 or len(m_vals) < 3:  # 样本量太小，检验无意义
            print(f"⚠️ {m}：样本量不足（Present={len(p_vals)}, Missing={len(m_vals)}）")
        else:
            # 检验方差齐性（Levene检验，对非正态数据更稳健）
            levene_p = stats.levene(p_vals, m_vals).pvalue
            # 方差齐→普通独立T检验，不齐→Welch's T检验（更适合数量不等的情况）
            _, p_val = stats.ttest_ind(p_vals, m_vals, equal_var=(levene_p > 0.05))
        
        # 输出结果
        sig = "⭐ 显著差异" if (not pd.isna(p_val)) and p_val < 0.05 else "不显著"
        p_val_str = f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"
        print(f"{m:<17} | {p_mean:.6f} | {m_mean:.6f} | {p_val_str} | {sig}")

def plot_research_results(df_overall, df_steps):
    if df_overall.empty: return
    
    # 创建 2x3 的画布，前 4 个为分布图，后 2 个为时序图
    fig = plt.figure(figsize=(20, 12))
    sns.set_style("whitegrid")

    # 1. 全局指标分布图
    metrics = ['mean', 'max', 'var', 'p90']
    for i, m in enumerate(metrics, 1):
        ax = fig.add_subplot(2, 3, i)
        sns.boxplot(x='group', y=m, data=df_overall, palette="Set2", ax=ax)
        ax.set_title(f'Overall {m.upper()} Comparison')

    # 2. 时序演化图 (Attention Temporal Flow)
    ax_time = fig.add_subplot(2, 3, 5)
    sns.lineplot(data=df_steps, x='step', y='mean', hue='group', marker='o', ax=ax_time)
    ax_time.set_title('Attention Mean Flow over Steps')
    ax_time.set_ylabel('Mean Absolute Intensity')

    # 3. 差异比演化图 (Gap Ratio Flow)
    ax_gap = fig.add_subplot(2, 3, 6)
    # 计算每步的 M/P Ratio
    step_means = df_steps.groupby(['step', 'group'])['mean'].mean().unstack()
    if 'Missing' in step_means.columns and 'Present' in step_means.columns:
        gap_ratio = step_means['Missing'] / step_means['Present']
        ax_gap.plot(gap_ratio.index, gap_ratio.values, color='red', linestyle='--', marker='x')
        ax_gap.axhline(1.0, color='black', alpha=0.3)
        ax_gap.set_title('M/P Gap Ratio Flow')
        ax_gap.set_ylabel('Ratio')
        ax_gap.set_xlabel('Step')

    plt.tight_layout()
    plt.savefig('attention_temporal_research.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    PATH = "/data/yulin/DK/Delta-K/logs/detailed_attention_flow_sdxl.jsonl"
    df_overall, df_steps = load_comprehensive_data(PATH)
    
    if not df_overall.empty:
        run_statistical_report(df_overall)
        plot_research_results(df_overall, df_steps)