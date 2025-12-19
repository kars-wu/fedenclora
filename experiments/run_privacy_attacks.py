"""
隐私攻击实验主脚本 - 修复版
评估FedEncLoRA、FedLoRFDP和FedLoRA-DP对各种隐私攻击的防御效果

使用真实数据集和有效的攻击方法
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import copy

# 导入自定义模块
from experiments.data_utils import (
    load_sst2_dataset, load_alpaca_dataset, load_wikitext_dataset,
    load_dataset_for_experiment, create_member_nonmember_split,
    partition_dataset_for_fl, insert_canary, create_dataloader, collate_fn
)
from experiments.llm_attacks import (
    LLMMembershipInferenceAttack,
    LLMAttributeInferenceAttack,
    LLMDataExtractionAttack,
    run_all_attacks,
    summarize_attack_results
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 模型配置
    model_path: str = "/data/wuhao/model/Qwen2.5-3B-Instruct"
    
    # 数据集配置
    dataset: str = "sst2"  # sst2, alpaca, wikitext
    num_train_samples: int = 2000
    num_member_samples: int = 300
    num_non_member_samples: int = 300
    max_length: int = 128
    
    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # 训练配置
    num_clients: int = 3
    num_fl_rounds: int = 5
    local_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # 防御配置
    dp_epsilon: float = 5.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    
    # FedLoRFDP配置（低秩分解+DP）
    lorf_rank: int = 4  # 低秩分解的秩
    
    # 攻击配置
    attack_batch_size: int = 4
    
    # 其他
    seed: int = 42
    device: str = "cuda:0"
    output_dir: str = "./attack_results"
    canary_text: str = "My secret password is: XYZ789ABC"


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, device: str) -> Tuple[nn.Module, AutoTokenizer]:
    """加载基础模型"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def add_lora_to_model(model: nn.Module, config: ExperimentConfig) -> nn.Module:
    """添加LoRA适配器"""
    print("Adding LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    desc: str = "Training"
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_with_dp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float,
    noise_multiplier: float,
    desc: str = "Training with DP"
) -> float:
    """带差分隐私的训练"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 添加噪声
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_multiplier * max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device,
                    dtype=param.grad.dtype
                )
                param.grad += noise
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def low_rank_decompose(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    低秩分解（SVD）
    将矩阵 W 分解为 L @ R，其中 L: (m, r), R: (r, n)
    """
    if len(tensor.shape) != 2:
        return None, None
    
    try:
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        # 取前rank个奇异值
        r = min(rank, len(S))
        L = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        R = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
        return L.to(tensor.dtype), R.to(tensor.dtype)
    except:
        return None, None


def train_with_lorf_dp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float,
    noise_multiplier: float,
    lorf_rank: int,
    desc: str = "Training with LoRF+DP"
) -> float:
    """
    FedLoRFDP: 低秩分解 + 差分隐私
    
    1. 计算梯度
    2. 对梯度进行低秩分解
    3. 在低秩分量上添加DP噪声（噪声量更小）
    4. 重构梯度并更新
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # 对每个参数的梯度进行低秩分解和加噪
        for name, param in model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                grad = param.grad
                original_shape = grad.shape
                
                # 只对2D梯度进行低秩分解
                if len(original_shape) == 2:
                    L, R = low_rank_decompose(grad, lorf_rank)
                    
                    if L is not None and R is not None:
                        # 在低秩分量上添加噪声（噪声量与低秩维度相关，更小）
                        noise_scale = noise_multiplier * max_grad_norm / np.sqrt(lorf_rank)
                        
                        noise_L = torch.normal(
                            mean=0, std=noise_scale,
                            size=L.shape, device=L.device, dtype=L.dtype
                        )
                        noise_R = torch.normal(
                            mean=0, std=noise_scale,
                            size=R.shape, device=R.device, dtype=R.dtype
                        )
                        
                        L_noisy = L + noise_L
                        R_noisy = R + noise_R
                        
                        # 重构梯度
                        param.grad = (L_noisy @ R_noisy).to(param.grad.dtype)
                    else:
                        # 分解失败，使用普通DP
                        noise = torch.normal(
                            mean=0, std=noise_multiplier * max_grad_norm,
                            size=grad.shape, device=grad.device, dtype=grad.dtype
                        )
                        param.grad += noise
                else:
                    # 非2D梯度使用普通DP
                    noise = torch.normal(
                        mean=0, std=noise_multiplier * max_grad_norm,
                        size=grad.shape, device=grad.device, dtype=grad.dtype
                    )
                    param.grad += noise
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def federated_training(
    model: nn.Module,
    client_dataloaders: List[DataLoader],
    config: ExperimentConfig,
    defense_type: str = "none"
) -> nn.Module:
    """
    模拟联邦训练
    
    Args:
        model: 模型
        client_dataloaders: 各客户端的数据加载器
        config: 配置
        defense_type: "none", "dp", "lorf_dp", "encryption"
    """
    print(f"\n{'='*60}")
    print(f"Federated Training - Defense: {defense_type.upper()}")
    print(f"{'='*60}")
    
    device = config.device
    model = model.to(device)
    
    # DP噪声参数
    noise_multiplier = 0.0
    if defense_type in ["dp", "lorf_dp"]:
        noise_multiplier = np.sqrt(2 * np.log(1.25 / config.dp_delta)) / config.dp_epsilon
        print(f"DP noise multiplier: {noise_multiplier:.4f}")
    
    for round_idx in range(config.num_fl_rounds):
        print(f"\n[Round {round_idx + 1}/{config.num_fl_rounds}]")
        
        round_losses = []
        
        for client_idx, client_loader in enumerate(client_dataloaders):
            # 创建优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            # 本地训练
            for epoch in range(config.local_epochs):
                if defense_type == "dp":
                    loss = train_with_dp(
                        model, client_loader, optimizer, device,
                        config.dp_max_grad_norm, noise_multiplier,
                        desc=f"Client {client_idx+1} Epoch {epoch+1}"
                    )
                elif defense_type == "lorf_dp":
                    loss = train_with_lorf_dp(
                        model, client_loader, optimizer, device,
                        config.dp_max_grad_norm, noise_multiplier,
                        config.lorf_rank,
                        desc=f"Client {client_idx+1} Epoch {epoch+1}"
                    )
                else:  # none 或 encryption
                    loss = train_one_epoch(
                        model, client_loader, optimizer, device,
                        desc=f"Client {client_idx+1} Epoch {epoch+1}"
                    )
            
            round_losses.append(loss)
            print(f"  Client {client_idx + 1}/{len(client_dataloaders)}: Final Loss = {loss:.4f}")
        
        print(f"  Round Average Loss: {np.mean(round_losses):.4f}")
    
    return model


def evaluate_model_utility(
    model: nn.Module,
    test_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """评估模型效用（perplexity）"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """运行完整实验"""
    set_seed(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载tokenizer
    print("\n" + "="*60)
    print("PRIVACY ATTACK EXPERIMENTS")
    print("="*60)
    
    _, tokenizer = load_model(config.model_path, config.device)
    
    # 加载数据集
    print(f"\nLoading {config.dataset} dataset...")
    train_dataset, test_dataset = load_dataset_for_experiment(
        config.dataset,
        tokenizer,
        max_length=config.max_length,
        num_samples=config.num_train_samples
    )
    
    # 创建成员/非成员划分
    member_dataset, non_member_dataset = create_member_nonmember_split(
        train_dataset,
        test_dataset,
        member_size=config.num_member_samples,
        non_member_size=config.num_non_member_samples
    )
    
    # 划分客户端数据（均匀划分）
    client_datasets = partition_dataset_for_fl(
        train_dataset,
        config.num_clients
    )
    
    # 在第一个客户端的数据中插入canary
    canary_client0_dataset, canary_text = insert_canary(
        client_datasets[0],  # 只在client0的数据中插入
        tokenizer,
        canary_text=config.canary_text,
        num_copies=10
    )
    
    # 创建数据加载器
    member_loader = create_dataloader(member_dataset, config.attack_batch_size, shuffle=False)
    non_member_loader = create_dataloader(non_member_dataset, config.attack_batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, config.attack_batch_size, shuffle=False)
    
    # 属性推理攻击数据
    attr_train_size = int(len(train_dataset) * 0.7)
    attr_train_loader = create_dataloader(
        Subset(train_dataset, range(attr_train_size)),
        config.attack_batch_size, shuffle=False
    )
    attr_test_loader = create_dataloader(
        Subset(train_dataset, range(attr_train_size, len(train_dataset))),
        config.attack_batch_size, shuffle=False
    )
    
    # 创建各客户端的数据加载器（均匀分布）
    client_loaders = [
        create_dataloader(canary_client0_dataset, config.batch_size, shuffle=True),  # Client 0带canary
        *[create_dataloader(ds, config.batch_size, shuffle=True) for ds in client_datasets[1:]]  # 其他客户端
    ]
    
    print(f"\nClient data distribution:")
    for i, loader in enumerate(client_loaders):
        print(f"  Client {i}: {len(loader.dataset)} samples")
    
    all_results = {}
    
    # ================================================================
    # 实验1: 无防御 (FedLoRA baseline)
    # ================================================================
    print("\n" + "="*60)
    print("Experiment 1: No Defense (FedLoRA)")
    print("="*60)
    
    base_model, _ = load_model(config.model_path, config.device)
    model_no_defense = add_lora_to_model(base_model, config)
    
    model_no_defense = federated_training(
        model_no_defense,
        client_loaders,
        config,
        defense_type="none"
    )
    
    utility_no_defense = evaluate_model_utility(model_no_defense, test_loader, config.device)
    print(f"Model Utility - Loss: {utility_no_defense['loss']:.4f}, PPL: {utility_no_defense['perplexity']:.2f}")
    
    attacks_no_defense = run_all_attacks(
        model_no_defense, tokenizer,
        member_loader, non_member_loader,
        attr_train_loader, attr_test_loader,
        config.device,
        defense_name="No Defense (FedLoRA)",
        canary_text=canary_text,
        client_loaders=client_loaders
    )
    
    all_results["no_defense"] = {
        "utility": utility_no_defense,
        "attacks": attacks_no_defense,
        "summary": summarize_attack_results(attacks_no_defense)
    }
    
    del model_no_defense, base_model
    torch.cuda.empty_cache()
    
    # ================================================================
    # 实验2: FedLoRA-DP (差分隐私)
    # ================================================================
    print("\n" + "="*60)
    print(f"Experiment 2: FedLoRA-DP (ε={config.dp_epsilon})")
    print("="*60)
    
    base_model, _ = load_model(config.model_path, config.device)
    model_dp = add_lora_to_model(base_model, config)
    
    model_dp = federated_training(
        model_dp,
        client_loaders,
        config,
        defense_type="dp"
    )
    
    utility_dp = evaluate_model_utility(model_dp, test_loader, config.device)
    print(f"Model Utility - Loss: {utility_dp['loss']:.4f}, PPL: {utility_dp['perplexity']:.2f}")
    
    attacks_dp = run_all_attacks(
        model_dp, tokenizer,
        member_loader, non_member_loader,
        attr_train_loader, attr_test_loader,
        config.device,
        defense_name=f"FedLoRA-DP (ε={config.dp_epsilon})",
        canary_text=canary_text,
        client_loaders=client_loaders
    )
    
    all_results["dp_defense"] = {
        "utility": utility_dp,
        "attacks": attacks_dp,
        "summary": summarize_attack_results(attacks_dp)
    }
    
    del model_dp, base_model
    torch.cuda.empty_cache()
    
    # ================================================================
    # 实验3: FedLoRFDP (低秩分解+差分隐私) - 论文提出的方法
    # ================================================================
    print("\n" + "="*60)
    print(f"Experiment 3: FedLoRFDP (rank={config.lorf_rank}, ε={config.dp_epsilon})")
    print("="*60)
    
    base_model, _ = load_model(config.model_path, config.device)
    model_lorf_dp = add_lora_to_model(base_model, config)
    
    model_lorf_dp = federated_training(
        model_lorf_dp,
        client_loaders,
        config,
        defense_type="lorf_dp"
    )
    
    utility_lorf_dp = evaluate_model_utility(model_lorf_dp, test_loader, config.device)
    print(f"Model Utility - Loss: {utility_lorf_dp['loss']:.4f}, PPL: {utility_lorf_dp['perplexity']:.2f}")
    
    attacks_lorf_dp = run_all_attacks(
        model_lorf_dp, tokenizer,
        member_loader, non_member_loader,
        attr_train_loader, attr_test_loader,
        config.device,
        defense_name=f"FedLoRFDP (r={config.lorf_rank}, ε={config.dp_epsilon})",
        canary_text=canary_text,
        client_loaders=client_loaders
    )
    
    all_results["lorf_dp_defense"] = {
        "utility": utility_lorf_dp,
        "attacks": attacks_lorf_dp,
        "summary": summarize_attack_results(attacks_lorf_dp)
    }
    
    del model_lorf_dp, base_model
    torch.cuda.empty_cache()
    
    # ================================================================
    # 实验4: FedEncLoRA (加密聚合) - 论文提出的方法
    # ================================================================
    print("\n" + "="*60)
    print("Experiment 4: FedEncLoRA (Encryption)")
    print("="*60)
    
    base_model, _ = load_model(config.model_path, config.device)
    model_enc = add_lora_to_model(base_model, config)
    
    # FedEncLoRA在训练过程中与普通FedLoRA相同
    # 但其安全性来自于加密聚合，服务器无法获取单个客户端的更新
    # 在攻击评估中，我们假设攻击者只能访问最终聚合的模型
    model_enc = federated_training(
        model_enc,
        client_loaders,
        config,
        defense_type="encryption"
    )
    
    utility_enc = evaluate_model_utility(model_enc, test_loader, config.device)
    print(f"Model Utility - Loss: {utility_enc['loss']:.4f}, PPL: {utility_enc['perplexity']:.2f}")
    
    # 注意：FedEncLoRA主要防御的是服务器端攻击（从单个客户端更新推断信息）
    # 对于最终模型的攻击（MIA、AIA），加密不能提供额外保护
    # FedEncLoRA的优势在于：
    # 1. 不会因为加噪而损失模型性能（与DP相比）
    # 2. 服务器无法区分或分析单个客户端的更新
    # 梯度泄露攻击可以展示这一优势
    attacks_enc = run_all_attacks(
        model_enc, tokenizer,
        member_loader, non_member_loader,
        attr_train_loader, attr_test_loader,
        config.device,
        defense_name="FedEncLoRA (Encryption)",
        canary_text=canary_text,
        client_loaders=None  # FedEncLoRA: 服务器无法访问单个客户端更新，设为None模拟这种情况
    )
    
    all_results["encryption_defense"] = {
        "utility": utility_enc,
        "attacks": attacks_enc,
        "summary": summarize_attack_results(attacks_enc)
    }
    
    # ================================================================
    # 结果汇总
    # ================================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\n📊 Model Utility Comparison:")
    print(f"  {'Defense':<30} {'Loss':>10} {'PPL':>10}")
    print(f"  {'-'*55}")
    for name, result in all_results.items():
        display_name = {
            "no_defense": "FedLoRA (No Defense)",
            "dp_defense": f"FedLoRA-DP (ε={config.dp_epsilon})",
            "lorf_dp_defense": f"FedLoRFDP (r={config.lorf_rank})",
            "encryption_defense": "FedEncLoRA"
        }.get(name, name)
        print(f"  {display_name:<30} {result['utility']['loss']:>10.4f} {result['utility']['perplexity']:>10.2f}")
    
    print("\n🔒 Privacy Attack Comparison:")
    print("  (对于MIA/AIA/Canary: 越低 = 防御越好)")
    print("  (对于Grad Leak: N/A表示攻击者无法访问)")
    print(f"  {'Defense':<25} {'MIA AUC':>10} {'AIA Acc':>10} {'Canary':>10} {'Grad Leak':>12}")
    print(f"  {'-'*75}")
    for name, result in all_results.items():
        display_name = {
            "no_defense": "FedLoRA (No Defense)",
            "dp_defense": f"FedLoRA-DP (ε={config.dp_epsilon})",
            "lorf_dp_defense": f"FedLoRFDP (r={config.lorf_rank})",
            "encryption_defense": "FedEncLoRA"
        }.get(name, name)
        summary = result['summary']
        grad_leak = summary.get('gradient_privacy_leakage', None)
        grad_leak_str = f"{grad_leak:.4f}" if grad_leak is not None else "N/A (加密)"
        print(f"  {display_name:<25} {summary.get('mia_best_auc', 0.5):>10.4f} "
              f"{summary.get('aia_accuracy', 0.5):>10.4f} "
              f"{summary.get('canary_exposure', 0.5):>10.4f} "
              f"{grad_leak_str:>12}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(config.output_dir, f"attack_results_{timestamp}.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {result_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Privacy Attack Experiments")
    
    parser.add_argument("--model-path", type=str, 
                       default="/data/wuhao/model/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="sst2",
                       choices=["sst2", "alpaca", "wikitext"])
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dp-epsilon", type=float, default=5.0)
    parser.add_argument("--lorf-rank", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="./attack_results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_path=args.model_path,
        dataset=args.dataset,
        num_train_samples=args.num_samples,
        num_clients=args.num_clients,
        num_fl_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        dp_epsilon=args.dp_epsilon,
        lorf_rank=args.lorf_rank,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    run_experiment(config)


if __name__ == "__main__":
    main()
