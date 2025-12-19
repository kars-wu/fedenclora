"""
数据工具函数 - 加载真实数据集用于隐私攻击实验
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from typing import Dict, List, Optional, Tuple, Callable
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import numpy as np
import random
import hashlib


class TextDataset(Dataset):
    """
    通用文本数据集，用于LLM训练和攻击实验
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        labels: Optional[List[int]] = None,
        attributes: Optional[List[int]] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.attributes = attributes
        
        # 预先tokenize以提高效率
        self.encodings = []
        for text in texts:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze()
            })
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "labels": self.encodings[idx]["input_ids"].clone(),  # LM任务
        }
        
        if self.attributes is not None:
            item["attribute"] = torch.tensor(self.attributes[idx])
        
        return item


class InstructionDataset(Dataset):
    """
    指令数据集，用于LLM微调
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        template: str = "qwen"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self._format_prompt(item)
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }
        
        if "attribute" in item:
            result["attribute"] = torch.tensor(item["attribute"])
        
        return result
    
    def _format_prompt(self, item: Dict) -> str:
        """格式化prompt"""
        instruction = item.get("instruction", item.get("question", ""))
        input_text = item.get("input", "")
        output = item.get("output", item.get("answer", ""))
        
        if self.template == "qwen":
            if input_text:
                prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:  # alpaca
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt


def load_wikitext_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_samples: int = 2000
) -> Tuple[Dataset, Dataset]:
    """
    加载WikiText-2数据集 - 经典的语言模型评估数据集
    """
    print("Loading WikiText-2 dataset...")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    
    # 过滤空行并分块
    train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 50]
    test_texts = [t for t in dataset["test"]["text"] if len(t.strip()) > 50]
    
    # 限制样本数量
    train_texts = train_texts[:num_samples]
    test_texts = test_texts[:min(num_samples // 4, len(test_texts))]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, tokenizer, max_length)
    
    print(f"WikiText - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def load_sst2_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    num_samples: int = 2000
) -> Tuple[Dataset, Dataset]:
    """
    加载SST-2情感分析数据集 - 用于属性推理攻击（情感作为敏感属性）
    """
    print("Loading SST-2 dataset...")
    
    dataset = load_dataset("sst2", trust_remote_code=True)
    
    # 处理训练集
    train_data = dataset["train"]
    train_texts = train_data["sentence"][:num_samples]
    train_labels = train_data["label"][:num_samples]
    
    # 处理验证集作为测试集
    test_data = dataset["validation"]
    test_texts = test_data["sentence"][:num_samples // 4]
    test_labels = test_data["label"][:num_samples // 4]
    
    # 创建带属性的数据集（情感作为属性）
    train_dataset = TextDataset(
        train_texts, tokenizer, max_length, 
        attributes=train_labels
    )
    test_dataset = TextDataset(
        test_texts, tokenizer, max_length,
        attributes=test_labels
    )
    
    print(f"SST-2 - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"  Positive samples: {sum(train_labels)}, Negative: {len(train_labels) - sum(train_labels)}")
    
    return train_dataset, test_dataset


def load_medical_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_samples: int = 1000
) -> Tuple[Dataset, Dataset]:
    """
    加载医疗相关数据集 - PubMedQA
    医疗领域数据更具敏感性，适合评估隐私攻击
    """
    print("Loading PubMedQA dataset...")
    
    try:
        dataset = load_dataset("pubmed_qa", "pqa_labeled", trust_remote_code=True)
        
        def format_qa(item):
            question = item["question"]
            context = " ".join(item["context"]["contexts"][:2])  # 取前两个context
            answer = item["long_answer"]
            return {
                "instruction": question,
                "input": context[:500],  # 限制长度
                "output": answer,
                "attribute": 0 if item["final_decision"] == "no" else 1
            }
        
        train_data = [format_qa(item) for item in dataset["train"]][:num_samples]
        
        # 分割训练和测试
        split_idx = int(len(train_data) * 0.8)
        
        train_dataset = InstructionDataset(train_data[:split_idx], tokenizer, max_length)
        test_dataset = InstructionDataset(train_data[split_idx:], tokenizer, max_length)
        
    except Exception as e:
        print(f"Failed to load PubMedQA: {e}")
        print("Using synthetic medical data instead...")
        train_dataset, test_dataset = create_synthetic_medical_data(tokenizer, max_length, num_samples)
    
    print(f"Medical - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def load_alpaca_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_samples: int = 2000
) -> Tuple[Dataset, Dataset]:
    """
    加载Alpaca指令数据集
    """
    print("Loading Alpaca dataset...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", trust_remote_code=True)
        data = list(dataset["train"])[:num_samples]
        
        # 添加领域属性（基于关键词分类）
        def assign_domain(item):
            text = (item.get("instruction", "") + item.get("output", "")).lower()
            if any(w in text for w in ["code", "program", "python", "function", "algorithm"]):
                return 0  # 编程
            elif any(w in text for w in ["write", "story", "poem", "essay", "creative"]):
                return 1  # 创作
            elif any(w in text for w in ["explain", "what is", "define", "describe"]):
                return 2  # 知识问答
            else:
                return 3  # 其他
        
        for item in data:
            item["attribute"] = assign_domain(item)
        
    except Exception as e:
        print(f"Warning: Could not load Alpaca dataset ({e}), using synthetic data")
        data = generate_synthetic_instruction_data(num_samples)
    
    # 划分训练集和测试集
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_dataset = InstructionDataset(train_data, tokenizer, max_length)
    test_dataset = InstructionDataset(test_data, tokenizer, max_length)
    
    print(f"Alpaca - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def create_synthetic_medical_data(
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_samples: int = 1000
) -> Tuple[Dataset, Dataset]:
    """创建合成医疗数据"""
    conditions = ["diabetes", "hypertension", "asthma", "arthritis", "depression"]
    symptoms = ["fatigue", "headache", "chest pain", "shortness of breath", "dizziness"]
    
    data = []
    for i in range(num_samples):
        condition = random.choice(conditions)
        symptom = random.choice(symptoms)
        data.append({
            "instruction": f"What should I do if I have {symptom} and {condition}?",
            "input": "",
            "output": f"Given your symptoms of {symptom} and history of {condition}, you should consult a healthcare provider. General recommendations include...",
            "attribute": conditions.index(condition) % 2  # 二分类属性
        })
    
    split_idx = int(len(data) * 0.8)
    train_dataset = InstructionDataset(data[:split_idx], tokenizer, max_length)
    test_dataset = InstructionDataset(data[split_idx:], tokenizer, max_length)
    
    return train_dataset, test_dataset


def generate_synthetic_instruction_data(num_samples: int = 2000) -> List[Dict]:
    """生成合成指令数据"""
    templates = [
        ("Write a Python function to {task}.", "def {task_name}():\n    # Implementation\n    pass", 0),
        ("Explain the concept of {topic} in simple terms.", "{topic} is a concept that...", 2),
        ("Write a short story about {subject}.", "Once upon a time, {subject}...", 1),
        ("What are the key differences between {a} and {b}?", "The main differences are...", 2),
    ]
    
    tasks = ["sort a list", "find maximum", "calculate factorial", "reverse string"]
    topics = ["machine learning", "blockchain", "quantum computing", "neural networks"]
    subjects = ["a brave knight", "a lost cat", "time travel", "an AI awakening"]
    
    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        if "{task}" in template[0]:
            task = random.choice(tasks)
            instruction = template[0].format(task=task)
            output = template[1].format(task_name=task.replace(" ", "_"))
        elif "{topic}" in template[0]:
            topic = random.choice(topics)
            instruction = template[0].format(topic=topic)
            output = template[1].format(topic=topic)
        elif "{subject}" in template[0]:
            subject = random.choice(subjects)
            instruction = template[0].format(subject=subject)
            output = template[1].format(subject=subject)
        else:
            instruction = template[0].format(a="A", b="B")
            output = template[1]
        
        data.append({
            "instruction": instruction,
            "input": "",
            "output": output + f" [Sample {i}]",
            "attribute": template[2]
        })
    
    return data


def create_member_nonmember_split(
    train_dataset: Dataset,
    hold_out_dataset: Dataset,
    member_size: int = 500,
    non_member_size: int = 500
) -> Tuple[Dataset, Dataset]:
    """
    创建成员/非成员数据划分，用于MIA攻击
    
    关键：成员数据来自训练集，非成员数据来自独立的hold-out集
    """
    # 从训练集采样成员数据
    member_indices = random.sample(range(len(train_dataset)), min(member_size, len(train_dataset)))
    member_dataset = Subset(train_dataset, member_indices)
    
    # 从hold-out集采样非成员数据
    non_member_indices = random.sample(range(len(hold_out_dataset)), min(non_member_size, len(hold_out_dataset)))
    non_member_dataset = Subset(hold_out_dataset, non_member_indices)
    
    print(f"Member/Non-member split: {len(member_dataset)}/{len(non_member_dataset)}")
    
    return member_dataset, non_member_dataset


def partition_dataset_for_fl(
    dataset: Dataset,
    num_clients: int,
    partition_type: str = "iid",
    alpha: float = 0.5
) -> List[Subset]:
    """
    为联邦学习划分数据集
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
    else:
        # Non-IID划分
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * num_samples).astype(int)
        proportions[-1] = num_samples - proportions[:-1].sum()
        
        random.shuffle(indices)
        client_indices = []
        start = 0
        for prop in proportions:
            client_indices.append(indices[start:start + prop])
            start += prop
    
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    
    print(f"Dataset partitioned into {num_clients} clients:")
    for i, ds in enumerate(client_datasets):
        print(f"  Client {i}: {len(ds)} samples")
    
    return client_datasets


def insert_canary(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    canary_text: str = "My secret password is: ABC123XYZ",
    num_copies: int = 10
) -> Tuple[Dataset, str]:
    """
    在数据集中插入Canary（金丝雀）用于数据提取攻击
    
    Returns:
        带有canary的数据集和canary文本
    """
    # 创建canary样本
    canary_enc = tokenizer(
        canary_text,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    
    class CanaryDataset(Dataset):
        def __init__(self, base_dataset, canary_encoding, num_canary):
            self.base_dataset = base_dataset
            self.canary_encoding = {
                "input_ids": canary_encoding["input_ids"].squeeze(),
                "attention_mask": canary_encoding["attention_mask"].squeeze(),
                "labels": canary_encoding["input_ids"].squeeze()
            }
            self.num_canary = num_canary
            self.canary_indices = set(random.sample(
                range(len(base_dataset) + num_canary), num_canary
            ))
        
        def __len__(self):
            return len(self.base_dataset) + self.num_canary
        
        def __getitem__(self, idx):
            if idx in self.canary_indices:
                result = dict(self.canary_encoding)
                # 添加默认attribute
                result["attribute"] = torch.tensor(0)
                return result
            # 调整索引
            base_idx = idx - sum(1 for i in self.canary_indices if i < idx)
            if base_idx >= len(self.base_dataset):
                base_idx = base_idx % len(self.base_dataset)
            item = self.base_dataset[base_idx]
            # 确保有attribute字段
            if isinstance(item, dict) and "attribute" not in item:
                item = dict(item)
                item["attribute"] = torch.tensor(0)
            return item
    
    canary_dataset = CanaryDataset(dataset, canary_enc, num_copies)
    
    return canary_dataset, canary_text


def collate_fn(batch):
    """DataLoader的collate函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    # 安全检查attribute字段
    if batch[0] is not None and "attribute" in batch[0]:
        attrs = []
        for item in batch:
            if "attribute" in item:
                attrs.append(item["attribute"])
        if len(attrs) == len(batch):
            result["attribute"] = torch.stack(attrs)
    
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def load_dataset_for_experiment(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    num_samples: int = 2000
) -> Tuple[Dataset, Dataset]:
    """
    根据名称加载数据集
    """
    loaders = {
        "wikitext": load_wikitext_dataset,
        "sst2": load_sst2_dataset,
        "medical": load_medical_dataset,
        "alpaca": load_alpaca_dataset,
    }
    
    if dataset_name.lower() not in loaders:
        print(f"Unknown dataset: {dataset_name}, using alpaca")
        dataset_name = "alpaca"
    
    return loaders[dataset_name.lower()](tokenizer, max_length, num_samples)
