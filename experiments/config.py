"""
实验配置文件
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""
    model_name_or_path: str = "/data/wuhao/model/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = False
    

@dataclass
class LoRAConfig:
    """LoRA配置"""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_clients: int = 4
    num_rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512


@dataclass
class DPConfig:
    """差分隐私配置"""
    epsilon_values: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0, 7.5, 10.0])
    delta: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass
class AttackConfig:
    """攻击实验配置"""
    # MIA配置
    mia_num_shadow_models: int = 2
    mia_shadow_epochs: int = 3
    mia_attack_epochs: int = 30
    
    # AIA配置
    aia_attack_epochs: int = 30
    
    # DRA配置
    dra_num_samples: int = 3
    dra_iterations: int = 300
    dra_lr: float = 0.1
    
    # 通用配置
    num_attack_samples: int = 100


@dataclass
class ExperimentConfig:
    """总体实验配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    
    # 设备配置
    device_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    main_device: str = "cuda:0"
    
    # 数据配置
    dataset_name: str = "commonsense_qa"  # 或 "alpaca", "dolly"
    num_samples_per_client: int = 500
    
    # 输出配置
    output_dir: str = "./outputs"
    seed: int = 42

