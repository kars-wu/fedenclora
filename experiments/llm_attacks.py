"""
针对大语言模型的隐私攻击实现 - 修复版
支持:
1. Membership Inference Attack (MIA) - 多种攻击方法
2. Attribute Inference Attack (AIA) - 基于embedding的攻击
3. Data Extraction Attack - Canary提取攻击
4. Gradient Leakage Attack - 针对FL的梯度泄露攻击
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy import stats
from tqdm import tqdm
import copy
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Membership Inference Attack (MIA)
# ============================================================================

class LLMMembershipInferenceAttack:
    """
    针对LLM的成员推理攻击 - 增强版
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0",
        reference_model: Optional[nn.Module] = None
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.reference_model = reference_model
        self.target_model.eval()
        
        if self.reference_model is not None:
            self.reference_model.eval()
    
    @torch.no_grad()
    def compute_sample_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: nn.Module
    ) -> float:
        """计算单个样本的loss"""
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss.item()
    
    @torch.no_grad()
    def compute_pertoken_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """计算每个token的loss"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss.view(shift_labels.size())
    
    @torch.no_grad()
    def compute_min_k_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: nn.Module,
        k_percent: float = 20.0
    ) -> float:
        """Min-k% Prob Attack"""
        per_token_loss = self.compute_pertoken_loss(input_ids, attention_mask, model)
        
        mask = attention_mask[..., 1:].float()
        per_token_loss = per_token_loss * mask
        
        valid_losses = per_token_loss[mask > 0]
        
        if len(valid_losses) == 0:
            return float('inf')
        
        k = max(1, int(len(valid_losses) * k_percent / 100))
        top_k_losses = torch.topk(valid_losses, k).values
        
        return top_k_losses.mean().item()
    
    def compute_metrics_batch(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        compute_min_k: bool = True
    ) -> Dict[str, List[float]]:
        """批量计算攻击指标"""
        losses = []
        perplexities = []
        min_k_probs = []
        
        model.eval()
        
        for batch in tqdm(dataloader, desc="Computing attack metrics", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            for i in range(input_ids.size(0)):
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1]
                
                loss = self.compute_sample_loss(single_input, single_mask, model)
                losses.append(loss)
                perplexities.append(np.exp(loss))
                
                if compute_min_k:
                    min_k = self.compute_min_k_prob(single_input, single_mask, model)
                    min_k_probs.append(min_k)
        
        return {
            "loss": losses,
            "perplexity": perplexities,
            "min_k_prob": min_k_probs if compute_min_k else []
        }
    
    def loss_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Loss-based Attack"""
        print("  Running loss-based attack...")
        
        member_metrics = self.compute_metrics_batch(member_loader, self.target_model, compute_min_k=False)
        non_member_metrics = self.compute_metrics_batch(non_member_loader, self.target_model, compute_min_k=False)
        
        member_losses = np.array(member_metrics["loss"])
        non_member_losses = np.array(non_member_metrics["loss"])
        
        all_losses = np.concatenate([member_losses, non_member_losses])
        all_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
        
        fpr, tpr, thresholds = roc_curve(all_labels, -all_losses)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]
        
        member_preds = (member_losses < optimal_threshold).astype(int)
        non_member_preds = (non_member_losses < optimal_threshold).astype(int)
        
        y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
        y_pred = np.concatenate([member_preds, non_member_preds])
        y_scores = np.concatenate([-member_losses, -non_member_losses])
        
        return self._compute_metrics(y_true, y_pred, y_scores, "loss")
    
    def perplexity_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Perplexity-based Attack"""
        print("  Running perplexity-based attack...")
        
        member_metrics = self.compute_metrics_batch(member_loader, self.target_model, compute_min_k=False)
        non_member_metrics = self.compute_metrics_batch(non_member_loader, self.target_model, compute_min_k=False)
        
        member_ppls = np.array(member_metrics["perplexity"])
        non_member_ppls = np.array(non_member_metrics["perplexity"])
        
        all_ppls = np.concatenate([member_ppls, non_member_ppls])
        all_labels = np.concatenate([np.ones(len(member_ppls)), np.zeros(len(non_member_ppls))])
        
        fpr, tpr, thresholds = roc_curve(all_labels, -all_ppls)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]
        
        member_preds = (member_ppls < optimal_threshold).astype(int)
        non_member_preds = (non_member_ppls < optimal_threshold).astype(int)
        
        y_true = np.concatenate([np.ones(len(member_ppls)), np.zeros(len(non_member_ppls))])
        y_pred = np.concatenate([member_preds, non_member_preds])
        y_scores = np.concatenate([-member_ppls, -non_member_ppls])
        
        return self._compute_metrics(y_true, y_pred, y_scores, "perplexity")
    
    def min_k_attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        k_percent: float = 20.0
    ) -> Dict[str, Any]:
        """Min-k% Prob Attack"""
        print(f"  Running Min-{k_percent}% prob attack...")
        
        member_metrics = self.compute_metrics_batch(member_loader, self.target_model, compute_min_k=True)
        non_member_metrics = self.compute_metrics_batch(non_member_loader, self.target_model, compute_min_k=True)
        
        member_scores = np.array(member_metrics["min_k_prob"])
        non_member_scores = np.array(non_member_metrics["min_k_prob"])
        
        all_scores = np.concatenate([member_scores, non_member_scores])
        all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        
        fpr, tpr, thresholds = roc_curve(all_labels, -all_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]
        
        member_preds = (member_scores < optimal_threshold).astype(int)
        non_member_preds = (non_member_scores < optimal_threshold).astype(int)
        
        y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        y_pred = np.concatenate([member_preds, non_member_preds])
        y_scores = np.concatenate([-member_scores, -non_member_scores])
        
        return self._compute_metrics(y_true, y_pred, y_scores, f"min_{k_percent}%")
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        attack_name: str
    ) -> Dict[str, Any]:
        """计算评估指标"""
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        tpr_at_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))] if len(fpr) > 0 else 0
        tpr_at_01fpr = tpr[np.argmin(np.abs(fpr - 0.001))] if len(fpr) > 0 else 0
        
        return {
            "attack_type": attack_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": auc,
            "advantage": accuracy_score(y_true, y_pred) - 0.5,
            "tpr@1%fpr": tpr_at_1fpr,
            "tpr@0.1%fpr": tpr_at_01fpr
        }
    
    def attack(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> Dict[str, Dict[str, Any]]:
        """执行所有MIA攻击"""
        results = {}
        
        results["loss"] = self.loss_attack(member_loader, non_member_loader)
        results["perplexity"] = self.perplexity_attack(member_loader, non_member_loader)
        results["min_k_10"] = self.min_k_attack(member_loader, non_member_loader, k_percent=10)
        results["min_k_20"] = self.min_k_attack(member_loader, non_member_loader, k_percent=20)
        
        return results


# ============================================================================
# Attribute Inference Attack (AIA)
# ============================================================================

class LLMAttributeInferenceAttack:
    """
    针对LLM的属性推理攻击
    从模型embedding推断训练数据的敏感属性
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        tokenizer: AutoTokenizer,
        num_attributes: int = 2,
        device: str = "cuda:0"
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.num_attributes = num_attributes
        self.device = device
        self.attack_classifier = None
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
        pooling: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提取模型的hidden state embeddings"""
        self.target_model.eval()
        embeddings = []
        attributes = []
        
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            
            if pooling == "mean":
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            elif pooling == "last":
                seq_lens = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lens]
            else:
                pooled = hidden_states[:, 0, :]
            
            embeddings.append(pooled.cpu().numpy())
            
            if "attribute" in batch:
                attributes.append(batch["attribute"].cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        attributes = np.concatenate(attributes, axis=0) if attributes else None
        
        return embeddings, attributes
    
    def train_attack_model(
        self,
        train_loader: DataLoader,
        method: str = "mlp",
        epochs: int = 50
    ):
        """训练属性推理攻击模型"""
        print(f"Training attribute inference attack ({method})...")
        
        embeddings, attributes = self.extract_embeddings(train_loader)
        
        if attributes is None:
            raise ValueError("Dataset must include 'attribute' field")
        
        if method == "lr":
            self.attack_classifier = LogisticRegression(max_iter=1000, random_state=42)
            self.attack_classifier.fit(embeddings, attributes)
        else:
            hidden_dim = embeddings.shape[1]
            self.attack_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, self.num_attributes)
            ).to(self.device)
            
            X = torch.FloatTensor(embeddings).to(self.device)
            y = torch.LongTensor(attributes).to(self.device)
            
            optimizer = torch.optim.Adam(self.attack_classifier.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            self.attack_classifier.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.attack_classifier(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    acc = (outputs.argmax(dim=1) == y).float().mean()
                    print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
    
    def attack(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """执行属性推理攻击"""
        if self.attack_classifier is None:
            raise ValueError("Attack model not trained")
        
        print("Running attribute inference attack...")
        
        embeddings, attributes = self.extract_embeddings(test_loader)
        
        if isinstance(self.attack_classifier, LogisticRegression):
            predictions = self.attack_classifier.predict(embeddings)
            probas = self.attack_classifier.predict_proba(embeddings)[:, 1]
        else:
            self.attack_classifier.eval()
            with torch.no_grad():
                X = torch.FloatTensor(embeddings).to(self.device)
                outputs = self.attack_classifier(X)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                probas = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        acc = accuracy_score(attributes, predictions)
        
        try:
            auc = roc_auc_score(attributes, probas)
        except:
            auc = 0.5
        
        return {
            "accuracy": acc,
            "precision": precision_score(attributes, predictions, average='weighted', zero_division=0),
            "recall": recall_score(attributes, predictions, average='weighted', zero_division=0),
            "f1": f1_score(attributes, predictions, average='weighted', zero_division=0),
            "auc": auc,
            "advantage": acc - (1.0 / self.num_attributes)
        }


# ============================================================================
# Data Extraction Attack - 修复版
# ============================================================================

class LLMDataExtractionAttack:
    """
    数据提取攻击 - 修复版
    
    评估模型是否记忆并泄露训练数据
    使用更合理的评估方法：比较canary与其他真实文本的PPL
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0"
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def compute_text_ppl(self, text: str) -> float:
        """计算文本的perplexity"""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        outputs = self.target_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=encoding["input_ids"]
        )
        
        return torch.exp(outputs.loss).item()
    
    @torch.no_grad()
    def compute_canary_exposure(
        self,
        canary_text: str,
        reference_texts: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        计算Canary Exposure度量 - 修复版
        
        使用真实文本作为参考，而不是随机token序列
        """
        print("Computing canary exposure...")
        
        self.target_model.eval()
        
        # 计算canary的perplexity
        canary_ppl = self.compute_text_ppl(canary_text)
        
        # 如果没有提供参考文本，生成一些相似长度的文本作为参考
        if reference_texts is None:
            # 生成一些"假"canary作为参考
            reference_texts = [
                "My secret password is: " + ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 9))
                for _ in range(num_samples)
            ]
        
        # 计算参考文本的PPL
        reference_ppls = []
        for text in tqdm(reference_texts[:num_samples], desc="Computing reference PPL", leave=False):
            try:
                ppl = self.compute_text_ppl(text)
                if not np.isinf(ppl) and not np.isnan(ppl):
                    reference_ppls.append(ppl)
            except:
                continue
        
        reference_ppls = np.array(reference_ppls)
        
        if len(reference_ppls) == 0:
            return {
                "canary_perplexity": canary_ppl,
                "reference_mean_ppl": float('inf'),
                "reference_std_ppl": 0,
                "exposure_rank": 0.5,
                "is_memorized": False
            }
        
        # 计算exposure rank: canary PPL 在参考分布中的排名（越低说明记忆程度越高）
        # rank = 比canary PPL更低的参考文本比例
        exposure_rank = np.sum(reference_ppls < canary_ppl) / len(reference_ppls)
        
        # 如果canary PPL 低于参考的均值-2*std，认为被记忆
        is_memorized = canary_ppl < (np.mean(reference_ppls) - 2 * np.std(reference_ppls))
        
        return {
            "canary_perplexity": canary_ppl,
            "reference_mean_ppl": float(np.mean(reference_ppls)),
            "reference_std_ppl": float(np.std(reference_ppls)),
            "exposure_rank": float(exposure_rank),  # 越低说明记忆程度越高
            "is_memorized": bool(is_memorized)
        }
    
    @torch.no_grad()
    def prompt_extraction(
        self,
        prefix_prompts: List[str],
        target_texts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Prompt-based数据提取攻击"""
        print("Running prompt extraction attack...")
        
        self.target_model.eval()
        
        results = {
            "exact_matches": 0,
            "partial_matches": 0,
            "similarities": [],
            "generations": []
        }
        
        for prefix, target in tqdm(zip(prefix_prompts, target_texts), total=len(prefix_prompts), leave=False):
            inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
            
            try:
                outputs = self.target_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = generated[len(prefix):]
                
            except Exception as e:
                generated = ""
            
            if target in generated:
                results["exact_matches"] += 1
            
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, target[:100], generated[:100]).ratio()
            results["similarities"].append(similarity)
            
            if similarity > 0.5:
                results["partial_matches"] += 1
            
            results["generations"].append({
                "prefix": prefix[:50],
                "target": target[:50],
                "generated": generated[:50],
                "similarity": similarity
            })
        
        results["exact_match_rate"] = results["exact_matches"] / len(prefix_prompts) if prefix_prompts else 0
        results["partial_match_rate"] = results["partial_matches"] / len(prefix_prompts) if prefix_prompts else 0
        results["mean_similarity"] = float(np.mean(results["similarities"])) if results["similarities"] else 0
        
        return results
    
    def attack(
        self,
        canary_text: Optional[str] = None,
        reference_texts: Optional[List[str]] = None,
        prefix_prompts: Optional[List[str]] = None,
        target_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """运行数据提取攻击"""
        results = {}
        
        if canary_text:
            results["canary"] = self.compute_canary_exposure(canary_text, reference_texts)
        
        if prefix_prompts and target_texts:
            results["extraction"] = self.prompt_extraction(prefix_prompts, target_texts)
        
        return results


# ============================================================================
# Gradient Leakage Attack (针对FL场景)
# ============================================================================

class GradientLeakageAttack:
    """
    梯度泄露攻击 - 针对联邦学习场景
    
    模拟服务器从客户端梯度/更新中推断信息的能力
    这是FedEncLoRA主要防御的攻击类型
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0"
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_gradient_similarity(
        self,
        batch1: Dict[str, torch.Tensor],
        batch2: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算两个batch产生的梯度相似度
        可用于判断服务器能否区分不同客户端的数据
        """
        self.target_model.train()
        
        # 计算batch1的梯度
        self.target_model.zero_grad()
        outputs1 = self.target_model(
            input_ids=batch1["input_ids"].to(self.device),
            attention_mask=batch1["attention_mask"].to(self.device),
            labels=batch1["labels"].to(self.device)
        )
        outputs1.loss.backward()
        
        grads1 = []
        for name, param in self.target_model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                grads1.append(param.grad.flatten().clone())
        
        if not grads1:
            return 0.0
        
        grad_vec1 = torch.cat(grads1)
        
        # 计算batch2的梯度
        self.target_model.zero_grad()
        outputs2 = self.target_model(
            input_ids=batch2["input_ids"].to(self.device),
            attention_mask=batch2["attention_mask"].to(self.device),
            labels=batch2["labels"].to(self.device)
        )
        outputs2.loss.backward()
        
        grads2 = []
        for name, param in self.target_model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                grads2.append(param.grad.flatten().clone())
        
        if not grads2:
            return 0.0
        
        grad_vec2 = torch.cat(grads2)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(grad_vec1.unsqueeze(0), grad_vec2.unsqueeze(0))
        
        return similarity.item()
    
    def evaluate_gradient_privacy(
        self,
        client_loaders: List[DataLoader],
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        评估梯度隐私泄露程度
        
        测量不同客户端梯度的可区分性
        FedEncLoRA应该使服务器无法区分不同客户端的更新
        """
        print("Evaluating gradient privacy leakage...")
        
        # 收集每个客户端的一些batch
        client_batches = []
        for loader in client_loaders:
            batches = []
            for i, batch in enumerate(loader):
                if i >= num_samples:
                    break
                batches.append(batch)
            client_batches.append(batches)
        
        # 计算同客户端内的梯度相似度
        intra_similarities = []
        for client_idx, batches in enumerate(client_batches):
            for i in range(len(batches) - 1):
                sim = self.compute_gradient_similarity(batches[i], batches[i+1])
                intra_similarities.append(sim)
        
        # 计算不同客户端间的梯度相似度
        inter_similarities = []
        for i in range(len(client_batches)):
            for j in range(i+1, len(client_batches)):
                if client_batches[i] and client_batches[j]:
                    sim = self.compute_gradient_similarity(
                        client_batches[i][0], 
                        client_batches[j][0]
                    )
                    inter_similarities.append(sim)
        
        intra_mean = np.mean(intra_similarities) if intra_similarities else 0
        inter_mean = np.mean(inter_similarities) if inter_similarities else 0
        
        # 可区分性 = 同客户端相似度 - 不同客户端相似度
        # 越高说明越容易区分不同客户端
        distinguishability = intra_mean - inter_mean
        
        return {
            "intra_client_similarity": float(intra_mean),
            "inter_client_similarity": float(inter_mean),
            "distinguishability": float(distinguishability),
            "privacy_leakage": float(max(0, distinguishability))  # 泄露程度
        }


# ============================================================================
# Unified Attack Runner
# ============================================================================

def run_all_attacks(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    attr_train_loader: DataLoader,
    attr_test_loader: DataLoader,
    device: str = "cuda:0",
    defense_name: str = "No Defense",
    reference_model: Optional[nn.Module] = None,
    canary_text: Optional[str] = None,
    client_loaders: Optional[List[DataLoader]] = None
) -> Dict[str, Dict]:
    """运行所有攻击实验"""
    print(f"\n{'='*60}")
    print(f"Running attacks against: {defense_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Membership Inference Attack
    print("\n[1] Membership Inference Attack")
    print("-" * 40)
    mia = LLMMembershipInferenceAttack(model, tokenizer, device, reference_model)
    results["mia"] = mia.attack(member_loader, non_member_loader)
    
    for attack_name, metrics in results["mia"].items():
        print(f"  {attack_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, "
              f"Advantage: {metrics['advantage']:.4f}")
    
    # 2. Attribute Inference Attack
    print("\n[2] Attribute Inference Attack")
    print("-" * 40)
    aia = LLMAttributeInferenceAttack(model, tokenizer, num_attributes=2, device=device)
    try:
        aia.train_attack_model(attr_train_loader, method="mlp", epochs=50)
        results["aia"] = aia.attack(attr_test_loader)
        print(f"  Accuracy: {results['aia']['accuracy']:.4f}, "
              f"AUC: {results['aia']['auc']:.4f}, "
              f"Advantage: {results['aia']['advantage']:.4f}")
    except Exception as e:
        print(f"  AIA failed: {e}")
        results["aia"] = {"accuracy": 0.5, "advantage": 0.0, "auc": 0.5}
    
    # 3. Data Extraction Attack
    print("\n[3] Data Extraction Attack")
    print("-" * 40)
    dea = LLMDataExtractionAttack(model, tokenizer, device)
    
    if canary_text:
        # 生成参考文本（与canary格式相似但内容不同）
        reference_texts = [
            f"My secret password is: {''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 9))}"
            for _ in range(50)
        ]
        results["extraction"] = dea.attack(canary_text=canary_text, reference_texts=reference_texts)
        canary_results = results["extraction"]["canary"]
        print(f"  Canary PPL: {canary_results['canary_perplexity']:.2f}")
        print(f"  Reference Mean PPL: {canary_results['reference_mean_ppl']:.2f}")
        print(f"  Exposure Rank: {canary_results['exposure_rank']:.4f} (lower = more memorized)")
        print(f"  Is Memorized: {canary_results['is_memorized']}")
    else:
        results["extraction"] = {"canary": {"exposure_rank": 0.5, "is_memorized": False}}
    
    # 4. Gradient Leakage Attack (仅用于评估FL场景)
    if client_loaders and len(client_loaders) >= 2:
        print("\n[4] Gradient Leakage Attack (FL-specific)")
        print("-" * 40)
        gla = GradientLeakageAttack(model, tokenizer, device)
        results["gradient_leakage"] = gla.evaluate_gradient_privacy(client_loaders, num_samples=5)
        print(f"  Intra-client similarity: {results['gradient_leakage']['intra_client_similarity']:.4f}")
        print(f"  Inter-client similarity: {results['gradient_leakage']['inter_client_similarity']:.4f}")
        print(f"  Privacy leakage: {results['gradient_leakage']['privacy_leakage']:.4f}")
    
    return results


def summarize_attack_results(results: Dict[str, Any]) -> Dict[str, float]:
    """汇总攻击结果为简洁指标"""
    summary = {}
    
    if "mia" in results:
        mia_aucs = [v["auc"] for v in results["mia"].values() if isinstance(v, dict)]
        summary["mia_best_auc"] = max(mia_aucs) if mia_aucs else 0.5
        summary["mia_best_advantage"] = max(v["advantage"] for v in results["mia"].values() if isinstance(v, dict))
    
    if "aia" in results:
        summary["aia_accuracy"] = results["aia"].get("accuracy", 0.5)
        summary["aia_auc"] = results["aia"].get("auc", 0.5)
    
    if "extraction" in results and "canary" in results["extraction"]:
        summary["canary_exposure"] = results["extraction"]["canary"].get("exposure_rank", 0.5)
        summary["canary_memorized"] = 1.0 if results["extraction"]["canary"].get("is_memorized", False) else 0.0
    
    if "gradient_leakage" in results:
        summary["gradient_privacy_leakage"] = results["gradient_leakage"].get("privacy_leakage", 0)
    
    return summary
