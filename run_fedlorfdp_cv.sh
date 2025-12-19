#!/bin/bash
# FedLoRFDP计算机视觉评估实验
# 针对审稿人意见，添加额外数据集评估

echo "======================================"
echo "FedLoRFDP CV Evaluation"
echo "======================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chatglm

cd /home/wuhao/fedenclora

# 创建输出目录
mkdir -p fedlorfdp_results

# ==============================
# ResNet-20 评估
# ==============================
echo ""
echo "===== ResNet-20 Evaluation ====="
echo ""

# 运行所有数据集评估 (IID + non-IID)
python -m experiments.fedlorfdp_cv_evaluation \
    --model resnet20 \
    --dataset all \
    --num-clients 3 \
    --num-rounds 50 \
    --local-epochs 3 \
    --lorf-rank 16 \
    --batch-size 256 \
    --dp-epsilon 8.0 \
    --output-dir ./fedlorfdp_results/resnet20

# ==============================
# 测试新数据集 
# ==============================
echo ""
echo "===== Quick Test on SVHN ====="
echo ""

python -m experiments.fedlorfdp_cv_evaluation \
    --model resnet20 \
    --dataset svhn \
    --partition iid \
    --num-clients 3 \
    --num-rounds 20 \
    --local-epochs 3 \
    --lorf-rank 16 \
    --dp-epsilon 8.0 \
    --device cuda:0 \
    --output-dir ./fedlorfdp_results

echo ""
echo "===== Quick Test on Fashion-MNIST ====="
echo ""

python -m experiments.fedlorfdp_cv_evaluation \
    --model resnet20 \
    --dataset fashion_mnist \
    --partition iid \
    --num-clients 3 \
    --num-rounds 20 \
    --local-epochs 3 \
    --lorf-rank 16 \
    --dp-epsilon 8.0 \
    --device cuda:0 \
    --output-dir ./fedlorfdp_results

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Results saved to: ./fedlorfdp_results/"
echo "======================================"

