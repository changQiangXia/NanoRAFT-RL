"""
干扰项检索器 - DistractorRetriever

功能：
1. 基于LlamaIndex检索与问题相关的chunks（正例）
2. 检索与问题无关的chunks作为干扰项（负例）
3. 支持多种干扰项注入策略

核心概念：
- Golden Chunk: 包含正确答案的chunk
- Distractor: 与问题相关但不包含答案的干扰chunk
- 目标：训练模型在噪声中识别正确信息
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from llama_index.schema import NodeWithScore


@dataclass
class RetrievedContext:
    """
    检索结果数据结构
    
    Attributes:
        golden_chunks: 包含答案的正确chunks
        distractor_chunks: 干扰项chunks
        all_chunks: 按特定顺序合并的所有chunks（用于input字段）
    """
    golden_chunks: List[str]
    distractor_chunks: List[str]
    all_chunks: List[str]
    golden_indices: List[int]  # 正确chunks在all_chunks中的位置


class DistractorRetriever:
    """
    干扰项检索器
    
    负责构建带干扰项的上下文，模拟真实RAG场景中的噪声。
    
    Attributes:
        index_retriever: LlamaIndex检索器
        num_distractors: 每个样本注入的干扰项数量
        shuffle: 是否打乱chunks顺序（增加难度）
    """
    
    # 干扰项注入策略
    STRATEGIES = ["random", "similar", "adversarial"]
    
    def __init__(
        self,
        index_retriever,
        num_distractors: int = 3,
        shuffle: bool = True,
        strategy: str = "random",
        random_seed: int = 42,
    ):
        """
        初始化检索器
        
        Args:
            index_retriever: LlamaIndex的检索器实例
            num_distractors: 干扰项数量（默认3个）
            shuffle: 是否打乱chunk顺序
            strategy: 干扰项选择策略 ["random", "similar", "adversarial"]
            random_seed: 随机种子
        """
        self.index_retriever = index_retriever
        self.num_distractors = num_distractors
        self.shuffle = shuffle
        self.strategy = strategy
        
        random.seed(random_seed)
        
        # 存储所有节点用于随机采样
        self._all_nodes = None
    
    def set_all_nodes(self, nodes: List):
        """
        设置全局节点池（用于随机采样干扰项）
        
        Args:
            nodes: 所有切分好的节点列表
        """
        self._all_nodes = nodes
    
    def retrieve_with_distractors(
        self,
        query: str,
        golden_chunk_text: Optional[str] = None,
    ) -> RetrievedContext:
        """
        检索带干扰项的上下文
        
        流程：
        1. 用query检索top-k相关chunks（其中部分可能是正确的）
        2. 根据策略选择干扰项
        3. 合并并打乱顺序
        
        Args:
            query: 查询问题
            golden_chunk_text: 已知的正确答案所在chunk（用于标记正例）
            
        Returns:
            RetrievedContext: 包含正负例的上下文结构
        """
        # 1. 检索相关chunks（默认返回更多，从中筛选）
        retrieved_nodes: List[NodeWithScore] = self.index_retriever.retrieve(query)
        
        # 2. 分离正例和候选干扰项
        golden_chunks = []
        candidate_distractors = []
        
        for node in retrieved_nodes:
            chunk_text = node.node.text
            
            # 如果提供了golden_chunk，判断是否匹配
            if golden_chunk_text and self._is_golden_chunk(chunk_text, golden_chunk_text):
                golden_chunks.append(chunk_text)
            else:
                candidate_distractors.append(chunk_text)
        
        # 如果没有明确标记golden_chunk，把top-1当作golden
        if not golden_chunk_text and retrieved_nodes:
            golden_chunks = [retrieved_nodes[0].node.text]
            candidate_distractors = [n.node.text for n in retrieved_nodes[1:]]
        
        # 3. 根据策略选择干扰项
        distractors = self._select_distractors(
            candidate_distractors, 
            query,
            needed=self.num_distractors
        )
        
        # 4. 合并并打乱
        all_chunks, golden_indices = self._merge_and_shuffle(
            golden_chunks, 
            distractors
        )
        
        return RetrievedContext(
            golden_chunks=golden_chunks,
            distractor_chunks=distractors,
            all_chunks=all_chunks,
            golden_indices=golden_indices,
        )
    
    def _is_golden_chunk(self, chunk_text: str, golden_text: str, threshold: float = 0.8) -> bool:
        """
        判断chunk是否为包含答案的golden chunk
        
        使用简单的文本重叠度计算，实际可用更复杂的语义匹配。
        
        Args:
            chunk_text: 待判断chunk
            golden_text: 标准答案chunk
            threshold: 相似度阈值
            
        Returns:
            bool: 是否为golden chunk
        """
        # 简单实现：计算字符重叠率
        chunk_set = set(chunk_text.lower().split())
        golden_set = set(golden_text.lower().split())
        
        if not chunk_set or not golden_set:
            return False
        
        overlap = len(chunk_set & golden_set) / len(chunk_set | golden_set)
        return overlap > threshold
    
    def _select_distractors(
        self,
        candidates: List[str],
        query: str,
        needed: int,
    ) -> List[str]:
        """
        根据策略选择干扰项
        
        Args:
            candidates: 候选干扰项列表
            query: 原始查询
            needed: 需要的干扰项数量
            
        Returns:
            List[str]: 选中的干扰项
        """
        if self.strategy == "random":
            # 随机策略：从所有节点中随机选择（与query可能无关）
            return self._random_distractors(needed)
        
        elif self.strategy == "similar":
            # 相似策略：从检索结果中选择低分但相关的
            # 这些chunks与query相关但不包含答案，迷惑性更强
            return candidates[:needed] if len(candidates) >= needed else candidates + self._random_distractors(needed - len(candidates))
        
        elif self.strategy == "adversarial":
            # 对抗策略：选择与golden_chunks相似的（需要更多计算）
            # 暂时退化为similar策略
            if len(candidates) >= needed:
                return candidates[:needed]
            else:
                # 候选不够时，用随机采样补充
                return candidates + self._random_distractors(needed - len(candidates))
        
        else:
            raise ValueError(f"未知的干扰项策略: {self.strategy}")
    
    def _random_distractors(self, needed: int) -> List[str]:
        """
        从全局节点池中随机采样干扰项
        
        Args:
            needed: 需要采样的数量
            
        Returns:
            List[str]: 随机干扰项
        """
        if not self._all_nodes:
            raise ValueError("请先调用set_all_nodes()设置节点池")
        
        # 随机采样
        sampled = random.sample(self._all_nodes, min(needed, len(self._all_nodes)))
        return [node.text for node in sampled]
    
    def _merge_and_shuffle(
        self,
        golden_chunks: List[str],
        distractors: List[str],
    ) -> Tuple[List[str], List[int]]:
        """
        合并正负例并打乱顺序
        
        Args:
            golden_chunks: 正确chunks
            distractors: 干扰项
            
        Returns:
            Tuple[List[str], List[int]]: (合并后的chunks, golden_chunk的索引列表)
        """
        all_chunks = golden_chunks + distractors
        
        if self.shuffle:
            # 记录原始golden索引
            original_golden_indices = list(range(len(golden_chunks)))
            
            # 创建索引列表并打乱
            indices = list(range(len(all_chunks)))
            random.shuffle(indices)
            
            # 根据打乱后的索引重排chunks
            shuffled_chunks = [all_chunks[i] for i in indices]
            
            # 计算新的golden索引
            new_golden_indices = [
                indices.index(i) for i in original_golden_indices
            ]
            
            return shuffled_chunks, sorted(new_golden_indices)
        else:
            golden_indices = list(range(len(golden_chunks)))
            return all_chunks, golden_indices
    
    def format_context(self, chunks: List[str], separator: str = "\n\n") -> str:
        """
        将chunks格式化为字符串（用于input字段）
        
        Args:
            chunks: chunk列表
            separator: 分隔符
            
        Returns:
            str: 格式化后的上下文
        """
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[Document {i}]\n{chunk}")
        return separator.join(formatted)


if __name__ == "__main__":
    # 使用示例
    from llama_index import VectorStoreIndex
    
    # 假设已有index
    # index = VectorStoreIndex(...)
    # retriever = DistractorRetriever(
    #     index_retriever=index.as_retriever(),
    #     num_distractors=3,
    #     strategy="random",
    # )
    # context = retriever.retrieve_with_distractors("什么是Selenium WebDriver?")
    pass
