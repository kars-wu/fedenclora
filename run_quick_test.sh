#!/bin/bash
# 快速测试版本 - 验证攻击代码是否正确

echo "快速测试 - 使用较少样本验证攻击效果"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

mkdir -p ./attack_results

python -m experiments.run_privacy_attacks \
    --model-path /data/wuhao/model/Qwen2.5-3B-Instruct \
    --dataset sst2 \
    --num-samples 500 \
    --num-clients 2 \
    --num-rounds 3 \
    --local-epochs 2 \
    --batch-size 8 \
    --dp-epsilon 5.0 \
    --device cuda:0 \
    --output-dir ./attack_results \
    --seed 42

