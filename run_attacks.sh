#!/bin/bash
# 隐私攻击实验运行脚本 - 完整版

echo ""
echo "######################################################################"
echo "#                    PRIVACY ATTACK EXPERIMENTS                      #"
echo "#          评估 FedLoRA / FedLoRA-DP / FedLoRFDP / FedEncLoRA         #"
echo "######################################################################"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p ./attack_results

# 运行实验
# 参数说明:
#   --num-samples: 训练样本数
#   --num-clients: 客户端数量
#   --num-rounds: 联邦训练轮数
#   --local-epochs: 每轮本地训练epoch数
#   --dp-epsilon: 差分隐私预算
#   --lorf-rank: FedLoRFDP的低秩分解秩

python -m experiments.run_privacy_attacks \
    --model-path /data/wuhao/model/Qwen2.5-3B-Instruct \
    --dataset sst2 \
    --num-samples 2000 \
    --num-clients 3 \
    --num-rounds 5 \
    --local-epochs 3 \
    --batch-size 4 \
    --dp-epsilon 5.0 \
    --lorf-rank 4 \
    --device cuda:0 \
    --output-dir ./attack_results \
    --seed 42

echo ""
echo "实验完成! 结果保存在 ./attack_results/"
echo ""
echo "评估的防御方案:"
echo "  1. FedLoRA (No Defense) - 基线，无防御"
echo "  2. FedLoRA-DP - 差分隐私防御"
echo "  3. FedLoRFDP - 低秩分解+差分隐私（论文方法）"
echo "  4. FedEncLoRA - 加密聚合（论文方法）"
