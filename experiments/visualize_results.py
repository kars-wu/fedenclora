"""
可视化脚本 - 生成论文中的攻击实验图表
针对 Qwen2.5-3B-Instruct 的实验结果
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Optional

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# 颜色方案
COLORS = {
    'No Defense (FedLoRA)': '#E74C3C',
    'FedLoRA-DP (ε=5)': '#3498DB',
    'FedEncLoRA': '#2ECC71',
    'FedLoRFDP (r=4, ε=5)': '#F39C12'
}


def load_results(result_file: str) -> Dict:
    """加载实验结果"""
    with open(result_file, 'r') as f:
        return json.load(f)


def plot_mia_results(results: Dict, output_dir: str):
    """
    绘制成员推理攻击结果
    """
    methods = list(results.keys())
    colors = [COLORS.get(m, '#808080') for m in methods]
    
    loss_acc = [results[m]['mia']['loss_attack']['accuracy'] for m in methods]
    loss_adv = [results[m]['mia']['loss_attack']['advantage'] for m in methods]
    ppl_acc = [results[m]['mia']['perplexity_attack']['accuracy'] for m in methods]
    ppl_adv = [results[m]['mia']['perplexity_attack']['advantage'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # 左图：攻击准确率
    bars1 = axes[0].bar(x - width/2, loss_acc, width, label='Loss-based', color='#3498DB', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, ppl_acc, width, label='Perplexity-based', color='#E74C3C', alpha=0.8)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random Guess')
    axes[0].set_ylabel('Attack Accuracy')
    axes[0].set_title('(a) MIA Attack Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(' (', '\n(') for m in methods], rotation=0, fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0.4, 0.75)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    # 右图：攻击优势
    bars3 = axes[1].bar(x - width/2, loss_adv, width, label='Loss-based', color='#3498DB', alpha=0.8)
    bars4 = axes[1].bar(x + width/2, ppl_adv, width, label='Perplexity-based', color='#E74C3C', alpha=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel('Attack Advantage')
    axes[1].set_title('(b) MIA Attack Advantage')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(' (', '\n(') for m in methods], rotation=0, fontsize=10)
    axes[1].legend(loc='upper right')
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mia_results.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'mia_results.png'), bbox_inches='tight')
    print(f"Saved: {output_dir}/mia_results.pdf")
    plt.close()


def plot_aia_results(results: Dict, output_dir: str):
    """
    绘制属性推理攻击结果
    """
    methods = list(results.keys())
    colors = [COLORS.get(m, '#808080') for m in methods]
    
    accuracy = [results[m]['aia']['accuracy'] for m in methods]
    advantage = [results[m]['aia']['advantage'] for m in methods]
    f1 = [results[m]['aia']['f1'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(methods))
    
    # 左图：准确率和F1
    width = 0.35
    bars1 = axes[0].bar(x - width/2, accuracy, width, label='Accuracy', color='#9B59B6', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, f1, width, label='F1 Score', color='#E67E22', alpha=0.8)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Random Guess')
    axes[0].set_ylabel('Score')
    axes[0].set_title('(a) AIA Attack Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0.3, 0.85)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # 右图：攻击优势
    bars3 = axes[1].bar(x, advantage, color=colors, alpha=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel('Attack Advantage')
    axes[1].set_title('(b) AIA Attack Advantage')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=10)
    
    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aia_results.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'aia_results.png'), bbox_inches='tight')
    print(f"Saved: {output_dir}/aia_results.pdf")
    plt.close()


def plot_gla_results(results: Dict, output_dir: str):
    """
    绘制梯度泄露攻击结果
    """
    methods = list(results.keys())
    colors = [COLORS.get(m, '#808080') for m in methods]
    
    token_acc = [results[m]['gla']['avg_token_accuracy'] for m in methods]
    text_sim = [results[m]['gla']['avg_text_similarity'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(methods))
    
    # Token准确率
    bars1 = axes[0].bar(x, token_acc, color=colors, alpha=0.8)
    axes[0].set_ylabel('Token Accuracy (↓ = Better Defense)')
    axes[0].set_title('(a) Gradient Leakage - Token Reconstruction')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=10)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # 文本相似度
    bars2 = axes[1].bar(x, text_sim, color=colors, alpha=0.8)
    axes[1].set_ylabel('Text Similarity (↓ = Better Defense)')
    axes[1].set_title('(b) Gradient Leakage - Text Reconstruction')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gla_results.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'gla_results.png'), bbox_inches='tight')
    print(f"Saved: {output_dir}/gla_results.pdf")
    plt.close()


def plot_combined_radar(results: Dict, output_dir: str):
    """
    绘制综合雷达图
    """
    categories = ['MIA\n(Loss)', 'MIA\n(PPL)', 'AIA\nAccuracy', 
                  'GLA\nToken', 'GLA\nText']
    
    methods = list(results.keys())
    
    # 提取数据
    data = {}
    for m in methods:
        data[m] = [
            results[m]['mia']['loss_attack']['accuracy'],
            results[m]['mia']['perplexity_attack']['accuracy'],
            results[m]['aia']['accuracy'],
            results[m]['gla']['avg_token_accuracy'],
            results[m]['gla']['avg_text_similarity']
        ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = [COLORS.get(m, '#808080') for m in methods]
    
    for method, color in zip(methods, colors):
        values = data[method] + data[method][:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_title('Privacy Attack Results (Lower = Better Defense)', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), bbox_inches='tight')
    print(f"Saved: {output_dir}/radar_comparison.pdf")
    plt.close()


def plot_all(result_file: str, output_dir: str):
    """
    生成所有图表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results...")
    data = load_results(result_file)
    results = data['results']
    
    print("\nGenerating figures...")
    print("-" * 40)
    
    plot_mia_results(results, output_dir)
    plot_aia_results(results, output_dir)
    plot_gla_results(results, output_dir)
    plot_combined_radar(results, output_dir)
    
    print("-" * 40)
    print(f"All figures saved to: {output_dir}/")


def generate_sample_results(output_file: str):
    """
    生成示例结果数据用于测试可视化
    """
    sample_results = {
        "config": {
            "model": "Qwen2.5-3B-Instruct",
            "num_clients": 4,
            "num_rounds": 5
        },
        "results": {
            "No Defense (FedLoRA)": {
                "mia": {
                    "loss_attack": {"accuracy": 0.623, "advantage": 0.123},
                    "perplexity_attack": {"accuracy": 0.658, "advantage": 0.158}
                },
                "aia": {"accuracy": 0.731, "advantage": 0.231, "f1": 0.712},
                "gla": {"avg_token_accuracy": 0.0856, "avg_text_similarity": 0.342}
            },
            "FedLoRA-DP (ε=5)": {
                "mia": {
                    "loss_attack": {"accuracy": 0.561, "advantage": 0.061},
                    "perplexity_attack": {"accuracy": 0.583, "advantage": 0.083}
                },
                "aia": {"accuracy": 0.598, "advantage": 0.098, "f1": 0.582},
                "gla": {"avg_token_accuracy": 0.0234, "avg_text_similarity": 0.187}
            },
            "FedEncLoRA": {
                "mia": {
                    "loss_attack": {"accuracy": 0.512, "advantage": 0.012},
                    "perplexity_attack": {"accuracy": 0.523, "advantage": 0.023}
                },
                "aia": {"accuracy": 0.528, "advantage": 0.028, "f1": 0.515},
                "gla": {"avg_token_accuracy": 0.0012, "avg_text_similarity": 0.089}
            },
            "FedLoRFDP (r=4, ε=5)": {
                "mia": {
                    "loss_attack": {"accuracy": 0.541, "advantage": 0.041},
                    "perplexity_attack": {"accuracy": 0.558, "advantage": 0.058}
                },
                "aia": {"accuracy": 0.562, "advantage": 0.062, "f1": 0.548},
                "gla": {"avg_token_accuracy": 0.0145, "avg_text_similarity": 0.156}
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_results, f, indent=2)
    
    print(f"Sample results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Visualize privacy attack results')
    parser.add_argument('--result-file', type=str, default=None,
                       help='Path to result JSON file')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--generate-sample', action='store_true',
                       help='Generate sample results for testing')
    
    args = parser.parse_args()
    
    if args.generate_sample:
        result_file = generate_sample_results('sample_results.json')
        plot_all(result_file, args.output_dir)
    elif args.result_file:
        plot_all(args.result_file, args.output_dir)
    else:
        print("Please provide --result-file or use --generate-sample")


if __name__ == "__main__":
    main()
