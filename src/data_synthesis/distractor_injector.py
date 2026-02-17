"""
干扰项注入器 - 修复递归深度超限问题

修复内容：
1. 简化 inject_with_retrieval 方法，避免复杂对象操作
2. 使用直接的列表操作替代可能导致递归的集合操作
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class DistractorDifficulty(Enum):
    """干扰项难度等级"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class InjectedContext:
    """注入干扰项后的上下文数据结构"""
    context_text: str
    golden_chunks: List[str]
    distractor_chunks: List[str]
    golden_positions: List[int]
    difficulty: str
    mixing_ratio: float
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)


class DistractorInjector:
    """干扰项注入器（修复版）"""
    
    # 预设配置
    PRESETS = {
        "easy": {
            "ratio_range": (0.3, 0.5),
            "description": "干扰项与问题无关，易于区分"
        },
        "medium": {
            "ratio_range": (0.5, 0.7),
            "description": "干扰项与问题相关但无答案，需要辨别"
        },
        "hard": {
            "ratio_range": (0.7, 0.9),
            "description": "干扰项与答案相似，容易混淆"
        },
    }
    
    def __init__(
        self,
        difficulty: DistractorDifficulty = DistractorDifficulty.MEDIUM,
        target_ratio: Optional[float] = None,
        shuffle: bool = True,
        random_seed: int = 42,
    ):
        if isinstance(difficulty, str):
            self.difficulty = DistractorDifficulty(difficulty.lower())
        else:
            self.difficulty = difficulty
        
        self.shuffle = shuffle
        random.seed(random_seed)
        
        if target_ratio is not None:
            self.target_ratio = target_ratio
        else:
            preset = self.PRESETS[self.difficulty.value]
            self.target_ratio = random.uniform(*preset["ratio_range"])
    
    def inject(
        self,
        golden_chunks: List[str],
        distractor_pool: List[str],
        num_chunks: Optional[int] = None,
    ) -> InjectedContext:
        """
        注入干扰项构建带噪上下文（基础方法，无递归风险）
        """
        if not golden_chunks:
            raise ValueError("必须提供至少一个golden_chunk")
        
        # 计算需要的干扰项数量
        if num_chunks is None:
            num_golden = len(golden_chunks)
            total_chunks = int(num_golden / (1 - self.target_ratio))
            num_distractors = total_chunks - num_golden
        else:
            num_distractors = num_chunks - len(golden_chunks)
        
        num_distractors = max(0, min(num_distractors, len(distractor_pool)))
        
        # 采样干扰项
        selected_distractors = random.sample(distractor_pool, num_distractors)
        
        # 合并
        all_chunks = list(golden_chunks) + list(selected_distractors)
        
        # 打乱顺序并记录位置
        if self.shuffle:
            # 创建带索引的列表
            indexed_chunks = list(enumerate(all_chunks))
            random.shuffle(indexed_chunks)
            
            # 提取打乱后的chunks
            shuffled_chunks = [chunk for _, chunk in indexed_chunks]
            
            # 计算golden的新位置（原始golden索引是 0 到 len(golden_chunks)-1）
            new_golden_positions = []
            for old_idx in range(len(golden_chunks)):
                for new_idx, (orig_idx, _) in enumerate(indexed_chunks):
                    if orig_idx == old_idx:
                        new_golden_positions.append(new_idx)
                        break
        else:
            shuffled_chunks = all_chunks
            new_golden_positions = list(range(len(golden_chunks)))
        
        # 格式化上下文
        context_text = self._format_context(shuffled_chunks)
        
        return InjectedContext(
            context_text=context_text,
            golden_chunks=list(golden_chunks),
            distractor_chunks=selected_distractors,
            golden_positions=sorted(new_golden_positions),
            difficulty=self.difficulty.value,
            mixing_ratio=num_distractors / len(shuffled_chunks) if shuffled_chunks else 0,
        )
    
    def inject_with_retrieval(
        self,
        query: str,
        golden_chunk: str,
        retriever,
        total_contexts: int = 5,
    ) -> InjectedContext:
        """
        基于检索结果注入干扰项（修复递归问题）
        
        关键修改：
        1. 直接调用检索器的底层方法，避免中间封装
        2. 使用简单的字符串比较替代复杂的对象比较
        3. 避免使用可能导致递归的集合操作
        """
        # 直接获取检索器的底层检索器
        index_retriever = retriever.index_retriever
        
        # 执行检索
        try:
            retrieved_nodes = index_retriever.retrieve(query)
        except Exception:
            retrieved_nodes = []
        
        # 分离golden chunks和候选干扰项
        golden_chunks = []
        candidate_distractors = []
        
        # 使用简单的字符串比较来判断是否是golden chunk
        golden_words = set(golden_chunk.lower().split())
        
        for node in retrieved_nodes:
            chunk_text = node.node.text
            
            # 计算文本相似度（简单版本）
            chunk_words = set(chunk_text.lower().split())
            if golden_words and chunk_words:
                overlap = len(golden_words & chunk_words) / len(golden_words | chunk_words)
                if overlap > 0.8:
                    if chunk_text not in golden_chunks:
                        golden_chunks.append(chunk_text)
                else:
                    candidate_distractors.append(chunk_text)
            else:
                candidate_distractors.append(chunk_text)
        
        # 如果没有找到golden，使用原始chunk
        if not golden_chunks:
            golden_chunks = [golden_chunk]
        
        # 确保候选干扰项不包含golden
        candidate_distractors = [
            c for c in candidate_distractors 
            if c not in golden_chunks
        ]
        
        # 计算需要的干扰项数量
        num_distractors_needed = total_contexts - len(golden_chunks)
        
        # 收集干扰项
        distractors = []
        
        # 首先使用检索到的候选
        for candidate in candidate_distractors:
            if len(distractors) >= num_distractors_needed:
                break
            if candidate not in distractors:
                distractors.append(candidate)
        
        # 如果候选不够，从全局节点池随机采样
        if len(distractors) < num_distractors_needed:
            remaining = num_distractors_needed - len(distractors)
            # 获取所有节点
            all_nodes = retriever._all_nodes if hasattr(retriever, '_all_nodes') else []
            
            # 过滤掉已经是golden或distractor的节点
            available = [
                n for n in all_nodes 
                if n.text not in golden_chunks and n.text not in distractors
            ]
            
            if available and remaining > 0:
                sampled = random.sample(available, min(remaining, len(available)))
                distractors.extend([n.text for n in sampled])
        
        # 合并并打乱
        all_chunks = golden_chunks + distractors
        original_golden_indices = list(range(len(golden_chunks)))
        
        if self.shuffle:
            indexed_chunks = list(enumerate(all_chunks))
            random.shuffle(indexed_chunks)
            shuffled_chunks = [chunk for _, chunk in indexed_chunks]
            
            # 计算新的golden位置
            new_golden_positions = []
            for old_idx in original_golden_indices:
                for new_idx, (orig_idx, _) in enumerate(indexed_chunks):
                    if orig_idx == old_idx:
                        new_golden_positions.append(new_idx)
                        break
        else:
            shuffled_chunks = all_chunks
            new_golden_positions = original_golden_indices
        
        # 格式化上下文
        context_text = self._format_context(shuffled_chunks)
        
        # 直接构造返回对象，避免复杂的中间操作
        return InjectedContext(
            context_text=context_text,
            golden_chunks=golden_chunks,
            distractor_chunks=distractors,
            golden_positions=sorted(new_golden_positions),
            difficulty=self.difficulty.value,
            mixing_ratio=len(distractors) / len(shuffled_chunks) if shuffled_chunks else 0,
        )
    
    def _format_context(self, chunks: List[str]) -> str:
        """格式化chunks为输入文本"""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            clean_chunk = " ".join(chunk.split())
            formatted.append(f"Document {i}: {clean_chunk}")
        return "\n\n".join(formatted)
    
    def get_difficulty_description(self) -> str:
        """获取当前难度的描述"""
        return self.PRESETS[self.difficulty.value]["description"]


# 保持与原始文件的兼容性，导出相同的类名
__all__ = ['DistractorInjector', 'InjectedContext', 'DistractorDifficulty']
