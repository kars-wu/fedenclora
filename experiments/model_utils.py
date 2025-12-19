"""
模型工具函数 - 加载Qwen2.5-3B-Instruct并配置LoRA
"""
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
import copy
from .config import ModelConfig, LoRAConfig


def load_model_and_tokenizer(
    config: ModelConfig,
    device: str = "cuda:0",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载Qwen2.5-3B-Instruct模型和tokenizer
    
    Args:
        config: 模型配置
        device: 设备
        load_in_8bit: 是否使用8bit量化
        load_in_4bit: 是否使用4bit量化
    
    Returns:
        模型和tokenizer
    """
    print(f"Loading model from {config.model_name_or_path}...")
    
    # 配置量化
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确定dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device if quantization_config else None,
        use_cache=config.use_cache
    )
    
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    if not quantization_config:
        model = model.to(device)
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def add_lora_adapters(
    model: AutoModelForCausalLM,
    lora_config: LoRAConfig
) -> PeftModel:
    """
    为模型添加LoRA适配器
    
    Args:
        model: 基础模型
        lora_config: LoRA配置
    
    Returns:
        带有LoRA的模型
    """
    print("Adding LoRA adapters...")
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA adapters added.")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def get_lora_state_dict(model: PeftModel) -> Dict[str, torch.Tensor]:
    """
    获取LoRA参数的state_dict
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict


def set_lora_state_dict(model: PeftModel, state_dict: Dict[str, torch.Tensor]):
    """
    设置LoRA参数
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])


def aggregate_lora_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    """
    聚合多个客户端的LoRA权重 (FedAvg)
    
    Args:
        client_weights: 客户端LoRA权重列表
        weights: 加权系数，默认平均
    
    Returns:
        聚合后的权重
    """
    if weights is None:
        weights = [1.0 / len(client_weights)] * len(client_weights)
    
    aggregated = {}
    for key in client_weights[0].keys():
        aggregated[key] = sum(
            w * client_weights[i][key] 
            for i, w in enumerate(weights)
        )
    
    return aggregated


def apply_dp_noise_to_lora(
    state_dict: Dict[str, torch.Tensor],
    epsilon: float,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    device: str = "cuda:0"
) -> Dict[str, torch.Tensor]:
    """
    对LoRA参数添加差分隐私噪声 (模拟DP-FedAvg)
    
    Args:
        state_dict: LoRA参数
        epsilon: 隐私预算
        delta: 隐私参数
        max_grad_norm: 裁剪范数
    
    Returns:
        加噪后的参数
    """
    import numpy as np
    
    sensitivity = 2 * max_grad_norm
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    noisy_state_dict = {}
    for name, param in state_dict.items():
        noise = torch.randn_like(param, device=device) * sigma
        noisy_state_dict[name] = param + noise
    
    return noisy_state_dict


def apply_low_rank_compression(
    state_dict: Dict[str, torch.Tensor],
    compression_rank: int = 4
) -> Dict[str, torch.Tensor]:
    """
    对LoRA参数应用低秩压缩 (模拟FedLoRFDP)
    
    Args:
        state_dict: LoRA参数
        compression_rank: 压缩秩
    
    Returns:
        压缩后的参数
    """
    compressed = {}
    for name, param in state_dict.items():
        if param.dim() == 2 and min(param.shape) > compression_rank:
            # SVD压缩
            U, S, Vh = torch.linalg.svd(param, full_matrices=False)
            compressed[name] = U[:, :compression_rank] @ torch.diag(S[:compression_rank]) @ Vh[:compression_rank, :]
        else:
            compressed[name] = param.clone()
    
    return compressed


class FedEncLoRASimulator:
    """
    FedEncLoRA模拟器 - 模拟加密聚合过程
    实际上加密聚合后结果与明文聚合相同，这里主要用于对比
    """
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_weights = []
    
    def encrypt_and_upload(self, client_id: int, lora_weights: Dict[str, torch.Tensor]):
        """模拟加密上传"""
        # 在实际FedEncLoRA中，这里会进行DMCFE加密
        # 由于加密聚合后解密结果与明文聚合相同，这里直接保存
        if len(self.client_weights) <= client_id:
            self.client_weights.append(lora_weights)
        else:
            self.client_weights[client_id] = lora_weights
    
    def secure_aggregate(self) -> Dict[str, torch.Tensor]:
        """模拟安全聚合"""
        # 在实际FedEncLoRA中，服务器无法获得单个客户端的权重
        # 只能获得聚合结果
        return aggregate_lora_weights(self.client_weights)
    
    def reset(self):
        """重置状态"""
        self.client_weights = []


class FedLoRFDPSimulator:
    """
    FedLoRFDP模拟器 - 低秩分解 + 差分隐私
    """
    
    def __init__(
        self, 
        num_clients: int,
        compression_rank: int = 4,
        epsilon: float = 5.0,
        delta: float = 1e-5
    ):
        self.num_clients = num_clients
        self.compression_rank = compression_rank
        self.epsilon = epsilon
        self.delta = delta
        self.client_weights = []
    
    def compress_and_perturb(
        self, 
        lora_weights: Dict[str, torch.Tensor],
        device: str = "cuda:0"
    ) -> Dict[str, torch.Tensor]:
        """压缩并添加噪声"""
        # 1. 低秩压缩
        compressed = apply_low_rank_compression(lora_weights, self.compression_rank)
        # 2. 添加DP噪声
        perturbed = apply_dp_noise_to_lora(
            compressed, self.epsilon, self.delta, device=device
        )
        return perturbed
    
    def upload(self, client_id: int, lora_weights: Dict[str, torch.Tensor], device: str = "cuda:0"):
        """上传处理后的权重"""
        processed = self.compress_and_perturb(lora_weights, device)
        if len(self.client_weights) <= client_id:
            self.client_weights.append(processed)
        else:
            self.client_weights[client_id] = processed
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """聚合"""
        return aggregate_lora_weights(self.client_weights)
    
    def reset(self):
        """重置状态"""
        self.client_weights = []

