"""
FedLoRFDP 计算机视觉评估实验
针对审稿人意见，添加额外数据集评估

模型:
- CLIP ViT-B/32 (全参数微调)
- ResNet-20 (从头训练)

原数据集:
- CIFAR-10, CIFAR-100, EMNIST

新增数据集:
- SVHN (Street View House Numbers)
- Fashion-MNIST
"""

import os
# 设置多进程启动方式
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import time
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import torch.multiprocessing as mp
from multiprocessing import Queue, Process


# ============================================================================
# 可视化工具
# ============================================================================

class TrainingVisualizer:
    """训练过程可视化"""
    
    def __init__(self, output_dir: str = "./fedlorfdp_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.history = {}  # {method_name: {"rounds": [], "accuracy": [], "loss": []}}
    
    def add_method(self, method_name: str):
        """添加新方法"""
        self.history[method_name] = {
            "rounds": [],
            "accuracy": [],
            "loss": []
        }
    
    def log(self, method_name: str, round_idx: int, accuracy: float, loss: float = 0):
        """记录一轮训练结果"""
        if method_name not in self.history:
            self.add_method(method_name)
        self.history[method_name]["rounds"].append(round_idx)
        self.history[method_name]["accuracy"].append(accuracy)
        self.history[method_name]["loss"].append(loss)
    
    def plot_training_curves(self, dataset_name: str, partition: str):
        """绘制训练曲线（包含Accuracy和Loss）"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        
        colors = {
            "FedAvg": "#2196F3",
            "DP-FedAvg": "#FF9800", 
            "FedLoRFDP": "#4CAF50"
        }
        markers = {"FedAvg": "o", "DP-FedAvg": "s", "FedLoRFDP": "^"}
        
        # 准确率曲线
        for method, data in self.history.items():
            color = colors.get(method, "#9C27B0")
            marker = markers.get(method, "d")
            ax1.plot(data["rounds"], data["accuracy"], 
                    label=method, color=color, marker=marker, 
                    markersize=5, linewidth=2, markevery=max(1, len(data["rounds"])//10))
        
        ax1.set_xlabel("Communication Round", fontsize=12)
        ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax1.set_title(f"Accuracy - {dataset_name.upper()} ({partition.upper()})", fontsize=14)
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Loss曲线
        for method, data in self.history.items():
            color = colors.get(method, "#9C27B0")
            marker = markers.get(method, "d")
            if data["loss"]:
                ax2.plot(data["rounds"], data["loss"], 
                        label=method, color=color, marker=marker, 
                        markersize=5, linewidth=2, markevery=max(1, len(data["rounds"])//10))
        
        ax2.set_xlabel("Communication Round", fontsize=12)
        ax2.set_ylabel("Test Loss", fontsize=12)
        ax2.set_title(f"Loss - {dataset_name.upper()} ({partition.upper()})", fontsize=14)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 最终准确率对比柱状图
        methods = list(self.history.keys())
        final_accs = [self.history[m]["accuracy"][-1] if self.history[m]["accuracy"] else 0 
                      for m in methods]
        
        bars = ax3.bar(methods, final_accs, color=[colors.get(m, "#9C27B0") for m in methods])
        ax3.set_ylabel("Final Accuracy (%)", fontsize=12)
        ax3.set_title("Final Accuracy Comparison", fontsize=14)
        ax3.set_ylim(0, 100)
        
        for bar, acc in zip(bars, final_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 最终Loss对比柱状图
        final_losses = [self.history[m]["loss"][-1] if self.history[m]["loss"] else 0 
                       for m in methods]
        
        bars2 = ax4.bar(methods, final_losses, color=[colors.get(m, "#9C27B0") for m in methods])
        ax4.set_ylabel("Final Loss", fontsize=12)
        ax4.set_title("Final Loss Comparison", fontsize=14)
        
        for bar, loss in zip(bars2, final_losses):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{loss:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"training_curve_{dataset_name}_{partition}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Training curve saved to: {filepath}")
    
    def plot_comparison_table(self, results: Dict[str, Any], dataset_name: str):
        """绘制对比表格图"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        
        # 准备表格数据
        columns = ["Method", "Accuracy", "δ-Acc", "Data Transfer", "Runtime (s)"]
        cell_data = []
        
        fedavg_acc = results.get("fedavg", {}).get("accuracy", 0)
        
        for method, data in results.items():
            if "error" in data:
                continue
            
            method_name = {
                "fedavg": "FedAvg",
                "dp_fedavg": "DP-FedAvg",
                "fedlorfdp": "FedLoRFDP"
            }.get(method, method)
            
            acc = data.get("accuracy", 0)
            delta = data.get("delta_acc", acc - fedavg_acc if method != "fedavg" else 0)
            transfer = data.get("data_transfer", 1.0)
            runtime = data.get("runtime", 0)
            
            cell_data.append([
                method_name,
                f"{acc:.2f}%",
                f"{delta:+.2f}%" if method != "fedavg" else "-",
                f"{transfer:.3f}×",
                f"{runtime:.1f}"
            ])
        
        table = ax.table(
            cellText=cell_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colColours=['#E3F2FD'] * len(columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        plt.title(f"FedLoRFDP Results - {dataset_name.upper()}", fontsize=14, fontweight='bold', pad=20)
        
        filename = f"results_table_{dataset_name}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📋 Results table saved to: {filepath}")
    
    def clear(self):
        """清除历史记录"""
        self.history = {}


# ============================================================================
# 配置
# ============================================================================

@dataclass
class FedLoRFDPConfig:
    """FedLoRFDP实验配置"""
    # 模型
    model_type: str = "resnet20"  # resnet20, clip_vit
    
    # 数据集
    dataset: str = "cifar10"  # cifar10, cifar100, emnist, svhn, fashion_mnist
    num_clients: int = 10
    
    # 多GPU配置
    num_gpus: int = 4  # 使用的GPU数量
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    use_parallel: bool = False  # 是否使用多GPU并行
    partition_type: str = "iid"  # iid, non_iid
    dirichlet_alpha: float = 0.2  # non-iid参数
    
    # 训练
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    # FedLoRFDP参数
    lorf_rank: int = 16  # 低秩分解的秩
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    
    # 其他
    seed: int = 42
    device: str = "cuda:0"
    output_dir: str = "./fedlorfdp_results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据集加载
# ============================================================================

def get_dataset(dataset_name: str, train: bool = True) -> Tuple[torch.utils.data.Dataset, int]:
    """
    获取数据集
    
    Returns:
        dataset, num_classes
    """
    # 数据变换
    if dataset_name in ["cifar10", "cifar100", "svhn", "stl10"]:
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:  # MNIST-like
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 100
    elif dataset_name == "svhn":
        split = 'train' if train else 'test'
        dataset = torchvision.datasets.SVHN(
            root='./data', split=split, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == "fashion_mnist":
        transform_fm = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),  # 转为3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=train, download=True, transform=transform_fm
        )
        num_classes = 10
    elif dataset_name == "emnist":
        transform_em = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.EMNIST(
            root='./data', split='balanced', train=train, download=True, transform=transform_em
        )
        num_classes = 47
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset, num_classes


def partition_data(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 0.2
) -> List[List[int]]:
    """
    划分数据到各客户端
    
    Returns:
        各客户端的样本索引列表
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    if partition_type == "iid":
        random.shuffle(indices)
        samples_per_client = num_samples // num_clients
        client_indices = [
            indices[i * samples_per_client:(i + 1) * samples_per_client]
            for i in range(num_clients)
        ]
    else:  # non-iid (Dirichlet分布)
        # 获取标签
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            
            # Dirichlet分布
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)
            proportions[-1] = len(class_indices) - proportions[:-1].sum()
            
            start = 0
            for client_id in range(num_clients):
                client_indices[client_id].extend(
                    class_indices[start:start + proportions[client_id]].tolist()
                )
                start += proportions[client_id]
    
    return client_indices


# ============================================================================
# 模型定义
# ============================================================================

class ResNet20(nn.Module):
    """ResNet-20 for CIFAR"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_model(model_type: str, num_classes: int, device: str) -> nn.Module:
    """获取模型"""
    if model_type == "resnet20":
        model = ResNet20(num_classes)
    elif model_type == "clip_vit":
        # 使用预训练的CLIP ViT-B/32
        try:
            import clip
            model, _ = clip.load("ViT-B/32", device=device)
            # 修改分类头
            model.visual.proj = None
            model.fc = nn.Linear(512, num_classes)
        except ImportError:
            print("CLIP not available, using ResNet-18 instead")
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(512, num_classes)
    else:
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
    
    return model.to(device)


# ============================================================================
# FedLoRFDP 核心算法
# ============================================================================

def low_rank_decompose(tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    低秩分解（Power Iteration方法）
    """
    if len(tensor.shape) != 2:
        return None, None
    
    m, n = tensor.shape
    r = min(rank, m, n)
    
    try:
        # 使用SVD进行低秩分解
        U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
        L = U[:, :r] @ torch.diag(torch.sqrt(S[:r]))
        R = torch.diag(torch.sqrt(S[:r])) @ Vh[:r, :]
        return L.to(tensor.dtype), R.to(tensor.dtype)
    except:
        return None, None


def add_dp_noise(tensor: torch.Tensor, noise_multiplier: float, max_norm: float) -> torch.Tensor:
    """添加差分隐私噪声"""
    noise = torch.normal(
        mean=0,
        std=noise_multiplier * max_norm,
        size=tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype
    )
    return tensor + noise


def fedlorfdp_client_update(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    local_epochs: int,
    lorf_rank: int,
    noise_multiplier: float,
    max_grad_norm: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    FedLoRFDP客户端更新
    
    1. 本地训练
    2. 计算模型更新
    3. 低秩分解
    4. 添加DP噪声
    """
    model.train()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(local_epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    # 计算模型更新 Δ = W_new - W_old（包括所有state，如BN的running stats）
    updates = {}
    current_state = model.state_dict()
    param_names = set(name for name, _ in model.named_parameters())
    
    for name in current_state.keys():
        if name not in initial_state:
            continue
            
        delta = current_state[name] - initial_state[name]
        
        # 只对可训练参数应用低秩分解和DP噪声
        if name in param_names:
            # 对2D参数进行低秩分解 + DP噪声
            if len(delta.shape) == 2 and min(delta.shape) > lorf_rank:
                L, R = low_rank_decompose(delta, lorf_rank)
                if L is not None and R is not None:
                    # 在低秩分量上添加噪声
                    noise_scale = noise_multiplier * max_grad_norm / np.sqrt(lorf_rank)
                    L_noisy = add_dp_noise(L, noise_scale / max_grad_norm, max_grad_norm)
                    R_noisy = add_dp_noise(R, noise_scale / max_grad_norm, max_grad_norm)
                    updates[name] = (L_noisy @ R_noisy).to(delta.dtype)
                else:
                    updates[name] = add_dp_noise(delta, noise_multiplier, max_grad_norm)
            else:
                updates[name] = add_dp_noise(delta, noise_multiplier, max_grad_norm)
        else:
            # BN的running stats等直接复制，不加噪声
            updates[name] = delta
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return updates, avg_loss


def fedavg_aggregate(
    global_model: nn.Module,
    client_updates: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    debug: bool = False
) -> None:
    """FedAvg聚合"""
    global_state = global_model.state_dict()
    
    updated_count = 0
    total_update_norm = 0.0
    
    for name in global_state.keys():
        if name in client_updates[0]:
            # 对于整数类型（如num_batches_tracked），直接取第一个客户端的值
            if not global_state[name].is_floating_point():
                global_state[name] = client_updates[0][name]
                updated_count += 1
            else:
                weighted_sum = sum(
                    w * updates[name] for w, updates in zip(client_weights, client_updates)
                )
                global_state[name] = global_state[name] + weighted_sum
                updated_count += 1
                total_update_norm += weighted_sum.norm().item()
    
    if debug:
        print(f"    [Debug] Updated {updated_count} params, total update norm: {total_update_norm:.6f}")
    
    global_model.load_state_dict(global_state)


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


# ============================================================================
# 多GPU并行训练（使用多进程）
# ============================================================================

def _worker_train_client(
    result_queue: Queue,
    client_id: int,
    global_state_cpu: Dict[str, torch.Tensor],
    dataset_indices: List[int],
    dataset_name: str,
    num_classes: int,
    model_type: str,
    gpu_id: int,
    training_type: str,
    noise_multiplier: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    lorf_rank: int,
    dp_max_grad_norm: float
):
    """
    工作进程：在指定GPU上训练单个客户端
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # 重新加载数据集（每个进程需要独立加载）
        train_dataset, _ = get_dataset(dataset_name, train=True)
        subset = Subset(train_dataset, dataset_indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # 创建模型
        if model_type == "resnet20":
            client_model = ResNet20(num_classes).to(device)
        else:
            client_model = resnet18(pretrained=False)
            client_model.fc = nn.Linear(512, num_classes)
            client_model = client_model.to(device)
        
        # 加载全局状态
        local_state = {k: v.to(device) for k, v in global_state_cpu.items()}
        client_model.load_state_dict(local_state)
        
        initial_state = {k: v.clone() for k, v in client_model.state_dict().items()}
        
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        client_model.train()
        
        # 训练
        for _ in range(local_epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if training_type == "dp":
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), dp_max_grad_norm)
                    for param in client_model.parameters():
                        if param.grad is not None:
                            param.grad += torch.normal(
                                0, noise_multiplier * dp_max_grad_norm,
                                size=param.grad.shape, device=param.grad.device
                            )
                
                optimizer.step()
        
        # 计算更新（包括所有state，移到CPU）
        updates = {}
        current_state = client_model.state_dict()
        for name in current_state.keys():
            if name in initial_state:
                updates[name] = (current_state[name] - initial_state[name]).cpu()
        
        weight = len(dataset_indices)
        
        # 清理
        del client_model, initial_state
        torch.cuda.empty_cache()
        
        result_queue.put((client_id, updates, weight))
        
    except Exception as e:
        result_queue.put((client_id, None, str(e)))


def parallel_client_training(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    config: 'FedLoRFDPConfig',
    num_classes: int,
    training_type: str = "fedavg",
    noise_multiplier: float = 0,
    client_indices_list: Optional[List[List[int]]] = None
) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    多GPU并行训练所有客户端
    """
    num_clients = len(client_loaders)
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    gpu_ids = config.gpu_ids[:num_gpus]
    
    # 获取全局模型状态（CPU）
    global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
    
    client_updates = [None] * num_clients
    client_weights = [0.0] * num_clients
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    
    # 获取每个客户端的数据索引
    if client_indices_list is None:
        client_indices_list = [list(range(len(loader.dataset))) for loader in client_loaders]
    
    # 分批处理（每批最多num_gpus个客户端并行）
    for batch_start in range(0, num_clients, num_gpus):
        batch_end = min(batch_start + num_gpus, num_clients)
        batch_size = batch_end - batch_start
        
        result_queue = mp.Queue()
        processes = []
        
        for i, client_id in enumerate(range(batch_start, batch_end)):
            gpu_id = gpu_ids[i % num_gpus]
            
            # 获取客户端数据集的索引
            if hasattr(client_loaders[client_id].dataset, 'indices'):
                indices = client_loaders[client_id].dataset.indices
            else:
                indices = list(range(len(client_loaders[client_id].dataset)))
            
            p = mp.Process(
                target=_worker_train_client,
                args=(
                    result_queue,
                    client_id,
                    global_state_cpu,
                    indices,
                    config.dataset,
                    num_classes,
                    config.model_type,
                    gpu_id,
                    training_type,
                    noise_multiplier,
                    config.local_epochs,
                    config.learning_rate,
                    config.weight_decay,
                    config.batch_size,
                    config.lorf_rank,
                    config.dp_max_grad_norm
                )
            )
            p.start()
            processes.append(p)
        
        # 收集结果
        for _ in range(batch_size):
            client_id, updates, weight = result_queue.get()
            if updates is not None:
                client_updates[client_id] = {k: v.to(config.device) for k, v in updates.items()}
                client_weights[client_id] = weight / total_samples
            else:
                print(f"  ⚠️ Client {client_id} failed: {weight}")
        
        # 等待进程结束
        for p in processes:
            p.join()
    
    return client_updates, client_weights


def multi_gpu_lorfdp_training(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    config: 'FedLoRFDPConfig',
    num_classes: int,
    noise_multiplier: float = 0
) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    多GPU轮询训练 - FedLoRFDP专用
    """
    num_clients = len(client_loaders)
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    gpu_ids = config.gpu_ids[:num_gpus]
    
    # 获取全局状态
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    
    client_updates = []
    client_weights = []
    criterion = nn.CrossEntropyLoss()
    
    for client_id, loader in enumerate(client_loaders):
        gpu_id = gpu_ids[client_id % num_gpus]
        device = f"cuda:{gpu_id}"
        
        # 创建新模型
        if config.model_type == "resnet20":
            client_model = ResNet20(num_classes).to(device)
        else:
            client_model = resnet18(pretrained=False)
            client_model.fc = nn.Linear(512, num_classes)
            client_model = client_model.to(device)
        
        # 加载全局权重
        local_state = {k: v.to(device) for k, v in global_state.items()}
        client_model.load_state_dict(local_state)
        
        # 创建新优化器
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
        
        # 使用FedLoRFDP更新
        updates, _ = fedlorfdp_client_update(
            client_model, loader, optimizer, criterion,
            device, config.local_epochs,
            config.lorf_rank, noise_multiplier, config.dp_max_grad_norm
        )
        
        # 移到主GPU
        updates = {k: v.to(config.device) for k, v in updates.items()}
        
        client_updates.append(updates)
        client_weights.append(len(loader.dataset) / total_samples)
        
        # 清理
        del client_model, optimizer
    
    return client_updates, client_weights


def multi_gpu_round_robin_training(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    config: 'FedLoRFDPConfig',
    num_classes: int,
    training_type: str = "fedavg",
    noise_multiplier: float = 0
) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    多GPU轮询训练 - 简单可靠的实现
    """
    num_clients = len(client_loaders)
    total_samples = sum(len(loader.dataset) for loader in client_loaders)
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    gpu_ids = config.gpu_ids[:num_gpus]
    
    # 获取全局状态
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    
    client_updates = []
    client_weights = []
    criterion = nn.CrossEntropyLoss()
    
    for client_id, loader in enumerate(client_loaders):
        # 选择GPU
        gpu_id = gpu_ids[client_id % num_gpus]
        device = f"cuda:{gpu_id}"
        
        # 每次创建新模型（确保干净状态）
        if config.model_type == "resnet20":
            client_model = ResNet20(num_classes).to(device)
        else:
            client_model = resnet18(pretrained=False)
            client_model.fc = nn.Linear(512, num_classes)
            client_model = client_model.to(device)
        
        # 加载全局权重
        local_state = {k: v.to(device) for k, v in global_state.items()}
        client_model.load_state_dict(local_state)
        
        # 保存初始状态
        initial_state = {k: v.clone() for k, v in client_model.state_dict().items()}
        
        # 创建新优化器（确保参数引用正确）
        optimizer = torch.optim.SGD(
            client_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
        
        client_model.train()
        
        # 记录训练前后的loss变化
        first_loss = None
        last_loss = None
        batch_count = 0
        
        for epoch in range(config.local_epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                
                if first_loss is None:
                    first_loss = loss.item()
                last_loss = loss.item()
                batch_count += 1
                
                loss.backward()
                
                if training_type == "dp":
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), config.dp_max_grad_norm)
                    for param in client_model.parameters():
                        if param.grad is not None:
                            param.grad.add_(torch.randn_like(param.grad) * noise_multiplier * config.dp_max_grad_norm)
                
                optimizer.step()
        
        # 第一个客户端打印训练进度
        if client_id == 0:
            print(f"    [Debug] Client 0: {batch_count} batches, loss {first_loss:.4f} -> {last_loss:.4f}")
        
        # 计算更新（包括所有state_dict中的参数，包括BN的running_mean/var）
        updates = {}
        update_norm = 0.0
        current_state = client_model.state_dict()
        for name in current_state.keys():
            if name in initial_state:
                delta = current_state[name] - initial_state[name]
                updates[name] = delta.to(config.device)
                # 只对浮点数计算范数
                if delta.is_floating_point():
                    update_norm += delta.norm().item()
        
        # 调试：第一个客户端打印更新范数
        if client_id == 0 and len(client_updates) == 0:
            print(f"    [Debug] Client 0 update norm: {update_norm:.6f}, num params: {len(updates)}")
        
        client_updates.append(updates)
        client_weights.append(len(loader.dataset) / total_samples)
        
        # 清理
        del client_model, optimizer, initial_state
    
    return client_updates, client_weights


# ============================================================================
# 主实验函数
# ============================================================================

def run_fedlorfdp_experiment(config: FedLoRFDPConfig) -> Dict[str, Any]:
    """运行FedLoRFDP实验"""
    set_seed(config.seed)
    
    print(f"\n{'='*60}")
    print(f"FedLoRFDP Evaluation")
    print(f"Model: {config.model_type}, Dataset: {config.dataset}")
    print(f"Partition: {config.partition_type}, Clients: {config.num_clients}")
    print(f"{'='*60}")
    
    # 加载数据集
    train_dataset, num_classes = get_dataset(config.dataset, train=True)
    test_dataset, _ = get_dataset(config.dataset, train=False)
    
    print(f"Dataset: {config.dataset}, Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 划分数据
    client_indices = partition_data(
        train_dataset, config.num_clients,
        config.partition_type, config.dirichlet_alpha
    )
    
    # 创建客户端数据加载器
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True, persistent_workers=True)
        client_loaders.append(loader)
        print(f"  Client: {len(indices)} samples")
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 计算噪声参数
    noise_multiplier = np.sqrt(2 * np.log(1.25 / config.dp_delta)) / config.dp_epsilon
    print(f"\nDP Parameters: ε={config.dp_epsilon}, δ={config.dp_delta}")
    print(f"Noise multiplier: {noise_multiplier:.4f}")
    
    # 检测可用GPU
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    print(f"🚀 Using {num_gpus} GPUs for parallel training: {config.gpu_ids[:num_gpus]}")
    
    results = {}
    
    # 初始化可视化工具
    visualizer = TrainingVisualizer(config.output_dir)
    
    # ========== 方法1: FedAvg (Baseline) ==========
    print(f"\n--- Running FedAvg (Baseline) [Using {num_gpus} GPUs] ---")
    model_fedavg = get_model(config.model_type, num_classes, config.device)
    visualizer.add_method("FedAvg")
    
    start_time = time.time()
    pbar = tqdm(range(config.num_rounds), desc="FedAvg", ncols=100)
    for round_idx in pbar:
        # 多GPU轮询训练客户端
        client_updates, client_weights = multi_gpu_round_robin_training(
            model_fedavg, client_loaders, config, num_classes,
            training_type="fedavg", noise_multiplier=0
        )
        
        # 聚合
        fedavg_aggregate(model_fedavg, client_updates, client_weights, debug=(round_idx == 0))
        
        # 每轮评估并记录
        acc, loss = evaluate(model_fedavg, test_loader, config.device)
        visualizer.log("FedAvg", round_idx + 1, acc, loss)
        pbar.set_postfix({"Acc": f"{acc:.1f}%", "Loss": f"{loss:.3f}"})
    
    fedavg_time = time.time() - start_time
    fedavg_acc, _ = evaluate(model_fedavg, test_loader, config.device)
    print(f"✓ FedAvg Final: Accuracy = {fedavg_acc:.2f}%, Time = {fedavg_time:.2f}s")
    
    results["fedavg"] = {
        "accuracy": fedavg_acc,
        "runtime": fedavg_time,
        "data_transfer": 1.0  # baseline
    }
    
    # ========== 方法2: DP-FedAvg ==========
    print(f"\n--- Running DP-FedAvg [Using {num_gpus} GPUs] ---")
    model_dpfedavg = get_model(config.model_type, num_classes, config.device)
    visualizer.add_method("DP-FedAvg")
    
    start_time = time.time()
    pbar = tqdm(range(config.num_rounds), desc="DP-FedAvg", ncols=100)
    for round_idx in pbar:
        # 多GPU轮询训练客户端
        client_updates, client_weights = multi_gpu_round_robin_training(
            model_dpfedavg, client_loaders, config, num_classes,
            training_type="dp", noise_multiplier=noise_multiplier
        )
        
        fedavg_aggregate(model_dpfedavg, client_updates, client_weights)
        
        # 每轮评估并记录
        acc, loss = evaluate(model_dpfedavg, test_loader, config.device)
        visualizer.log("DP-FedAvg", round_idx + 1, acc, loss)
        pbar.set_postfix({"Acc": f"{acc:.1f}%", "Loss": f"{loss:.3f}"})
    
    dpfedavg_time = time.time() - start_time
    dpfedavg_acc, _ = evaluate(model_dpfedavg, test_loader, config.device)
    print(f"✓ DP-FedAvg Final: Accuracy = {dpfedavg_acc:.2f}%, Time = {dpfedavg_time:.2f}s")
    
    results["dp_fedavg"] = {
        "accuracy": dpfedavg_acc,
        "delta_acc": dpfedavg_acc - fedavg_acc,
        "runtime": dpfedavg_time,
        "data_transfer": 1.0
    }
    
    # ========== 方法3: FedLoRFDP ==========
    print(f"\n--- Running FedLoRFDP (rank={config.lorf_rank}) [Using {num_gpus} GPUs] ---")
    model_fedlorfdp = get_model(config.model_type, num_classes, config.device)
    visualizer.add_method("FedLoRFDP")
    
    # 计算压缩比
    total_params = sum(p.numel() for p in model_fedlorfdp.parameters())
    compressed_params = 0
    for name, param in model_fedlorfdp.named_parameters():
        if len(param.shape) == 2:
            m, n = param.shape
            r = min(config.lorf_rank, m, n)
            compressed_params += r * (m + n)
        else:
            compressed_params += param.numel()
    compression_ratio = compressed_params / total_params
    print(f"Compression ratio: {compression_ratio:.4f}")
    
    start_time = time.time()
    pbar = tqdm(range(config.num_rounds), desc="FedLoRFDP", ncols=100)
    for round_idx in pbar:
        # 多GPU轮询训练客户端（FedLoRFDP需要特殊处理）
        client_updates, client_weights = multi_gpu_lorfdp_training(
            model_fedlorfdp, client_loaders, config, num_classes,
            noise_multiplier=noise_multiplier
        )
        
        fedavg_aggregate(model_fedlorfdp, client_updates, client_weights)
        
        # 每轮评估并记录
        acc, loss = evaluate(model_fedlorfdp, test_loader, config.device)
        visualizer.log("FedLoRFDP", round_idx + 1, acc, loss)
        pbar.set_postfix({"Acc": f"{acc:.1f}%", "Loss": f"{loss:.3f}"})
    
    fedlorfdp_time = time.time() - start_time
    fedlorfdp_acc, _ = evaluate(model_fedlorfdp, test_loader, config.device)
    print(f"✓ FedLoRFDP Final: Accuracy = {fedlorfdp_acc:.2f}%, Time = {fedlorfdp_time:.2f}s")
    
    results["fedlorfdp"] = {
        "accuracy": fedlorfdp_acc,
        "delta_acc": fedlorfdp_acc - fedavg_acc,
        "runtime": fedlorfdp_time,
        "data_transfer": compression_ratio,
        "compression_ratio": compression_ratio
    }
    
    # ========== 生成可视化 ==========
    print(f"\n📊 Generating visualizations...")
    visualizer.plot_training_curves(config.dataset, config.partition_type)
    visualizer.plot_comparison_table(results, config.dataset)
    
    # ========== 结果汇总 ==========
    print(f"\n{'='*60}")
    print("📈 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset}, Model: {config.model_type}")
    print(f"Partition: {config.partition_type}")
    print(f"\n{'Method':<15} {'Accuracy':>10} {'δ-Acc':>10} {'Transfer':>12} {'Runtime':>10}")
    print("-" * 60)
    print(f"{'FedAvg':<15} {results['fedavg']['accuracy']:>10.2f}% {'-':>10} "
          f"{results['fedavg']['data_transfer']:>12.3f} {results['fedavg']['runtime']:>10.2f}s")
    print(f"{'DP-FedAvg':<15} {results['dp_fedavg']['accuracy']:>10.2f}% "
          f"{results['dp_fedavg']['delta_acc']:>+10.2f}% "
          f"{results['dp_fedavg']['data_transfer']:>12.3f} {results['dp_fedavg']['runtime']:>10.2f}s")
    print(f"{'FedLoRFDP':<15} {results['fedlorfdp']['accuracy']:>10.2f}% "
          f"{results['fedlorfdp']['delta_acc']:>+10.2f}% "
          f"{results['fedlorfdp']['data_transfer']:>12.3f} {results['fedlorfdp']['runtime']:>10.2f}s")
    
    return results


def run_all_datasets(config: FedLoRFDPConfig) -> Dict[str, Dict]:
    """运行所有数据集的评估"""
    datasets = ["cifar10", "cifar100", "svhn", "fashion_mnist"]
    partitions = ["iid", "non_iid"]
    
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        for partition in partitions:
            print(f"\n\n{'#'*70}")
            print(f"Dataset: {dataset}, Partition: {partition}")
            print(f"{'#'*70}")
            
            config.dataset = dataset
            config.partition_type = partition
            
            try:
                results = run_fedlorfdp_experiment(config)
                all_results[dataset][partition] = results
            except Exception as e:
                print(f"Error: {e}")
                all_results[dataset][partition] = {"error": str(e)}
    
    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(config.output_dir, f"fedlorfdp_cv_results_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to {result_file}")
    
    # 生成LaTeX表格
    generate_latex_table(all_results, config)
    
    return all_results


def generate_latex_table(results: Dict[str, Dict], config: FedLoRFDPConfig):
    """生成LaTeX表格"""
    print("\n\n" + "="*60)
    print("LaTeX Table:")
    print("="*60)
    
    print(r"""
\begin{table*}[ht]
    \caption{Evaluation of FedLoRFDP for """ + config.model_type + r""".}
    \centering
    \begin{tabular}{cccccccc}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Method} & \multicolumn{2}{c}{IID ($\gamma \approx 1.0$)} & \multicolumn{2}{c}{non-IID ($\gamma \approx 0.2$)} & Data Transfer & Runtime \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & FedAvg Acc. & $\delta$-Acc. & FedAvg Acc. & $\delta$-Acc. & ($\times$ FedAvg) & (s) \\
\midrule""")
    
    for dataset in results.keys():
        if "iid" in results[dataset] and "non_iid" in results[dataset]:
            iid = results[dataset]["iid"]
            noniid = results[dataset]["non_iid"]
            
            if "error" not in iid and "error" not in noniid:
                fedavg_iid = iid.get("fedavg", {}).get("accuracy", 0)
                fedavg_noniid = noniid.get("fedavg", {}).get("accuracy", 0)
                
                # FedLoRF行 (仅压缩)
                print(f"\\multirow{{3}}{{*}}{{{dataset.upper()}}} & FedLoRF & & - & & - & "
                      f"{iid.get('fedlorfdp', {}).get('compression_ratio', 0):.3f} & - \\\\")
                
                # DP-FedAvg行
                print(f"& DP-FedAvg & {fedavg_iid:.2f} & "
                      f"{iid.get('dp_fedavg', {}).get('delta_acc', 0):+.2f} & "
                      f"{fedavg_noniid:.2f} & "
                      f"{noniid.get('dp_fedavg', {}).get('delta_acc', 0):+.2f} & "
                      f"1.0 & {iid.get('dp_fedavg', {}).get('runtime', 0):.2f} \\\\")
                
                # FedLoRFDP行
                print(f"& FedLoRFDP & & "
                      f"{iid.get('fedlorfdp', {}).get('delta_acc', 0):+.2f} & & "
                      f"{noniid.get('fedlorfdp', {}).get('delta_acc', 0):+.2f} & "
                      f"{iid.get('fedlorfdp', {}).get('compression_ratio', 0):.3f} & "
                      f"{iid.get('fedlorfdp', {}).get('runtime', 0):.2f} \\\\")
                print("\\midrule")
    
    print(r"""\bottomrule
\end{tabular}
\end{table*}
""")


def main():
    # 设置多进程启动方式（必须在main中设置）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="FedLoRFDP CV Evaluation")
    
    parser.add_argument("--model", type=str, default="resnet20",
                       choices=["resnet20", "clip_vit"])
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["all", "cifar10", "cifar100", "svhn", "fashion_mnist", "emnist"])
    parser.add_argument("--partition", type=str, default="iid",
                       choices=["iid", "non_iid"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--lorf-rank", type=int, default=16)
    parser.add_argument("--dp-epsilon", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="./fedlorfdp_results")
    parser.add_argument("--num-gpus", type=int, default=4,
                       help="Number of GPUs for parallel training")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3",
                       help="Comma-separated GPU IDs to use")
    parser.add_argument("--parallel", action="store_true", default=False,
                       help="Use multi-GPU parallel training (may have issues)")
    parser.add_argument("--sequential", action="store_true", default=True,
                       help="Use sequential training (more stable)")
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    
    config = FedLoRFDPConfig(
        model_type=args.model,
        dataset=args.dataset,
        partition_type=args.partition,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        num_gpus=args.num_gpus,
        gpu_ids=gpu_ids,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lorf_rank=args.lorf_rank,
        dp_epsilon=args.dp_epsilon,
        device=args.device,
        output_dir=args.output_dir
    )
    
    if args.dataset == "all":
        run_all_datasets(config)
    else:
        run_fedlorfdp_experiment(config)


if __name__ == "__main__":
    main()

