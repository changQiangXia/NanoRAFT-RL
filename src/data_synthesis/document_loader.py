"""
文档加载器 - 基于LlamaIndex

功能：
1. 加载本地PDF/Markdown/Word文档
2. 智能切分长文档为Chunks
3. 构建向量索引供后续检索使用
"""

import os
from pathlib import Path
from typing import List, Optional

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Document,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI as LlamaOpenAI


class DocumentLoader:
    """
    文档加载器类
    
    负责从本地目录加载各类文档，并构建向量索引。
    考虑到4GB VRAM限制，默认使用轻量级Embedding模型。
    
    Attributes:
        data_dir: 原始文档存放目录
        chunk_size: 文档切分大小
        chunk_overlap: 切分重叠大小
        index: 构建好的向量索引
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        初始化文档加载器
        
        Args:
            data_dir: 原始文档目录路径
            chunk_size: 每个chunk的token数（推荐512-1024）
            chunk_overlap: chunk间重叠token数（防止信息割裂）
            embedding_model: 使用的Embedding模型名称
                          默认使用MiniLM (22MB)，适合4GB显存
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        self.index: Optional[VectorStoreIndex] = None
        self.documents: List[Document] = []
        self.nodes = []
        
        # 初始化Embedding模型（轻量级，CPU运行即可）
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            device="cpu",  # 4GB显存紧张，Embedding放CPU
        )
        
    def load_documents(self, required_exts: Optional[List[str]] = None) -> List[Document]:
        """
        从data_dir加载所有文档
        
        Args:
            required_exts: 限制文件后缀，如 [".md", ".pdf"]
                          None表示加载所有支持的格式
        
        Returns:
            List[Document]: 加载的文档列表
            
        Raises:
            FileNotFoundError: 当data_dir不存在时
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                f"请手动将文档放入该目录后再运行"
            )
        
        print(f"[DocumentLoader] 正在从 {self.data_dir} 加载文档...")
        
        reader = SimpleDirectoryReader(
            input_dir=str(self.data_dir),
            required_exts=required_exts,
            recursive=True,  # 递归子目录
        )
        
        self.documents = reader.load_data()
        print(f"[DocumentLoader] 成功加载 {len(self.documents)} 个文档")
        
        # 打印文档统计信息
        total_chars = sum(len(doc.text) for doc in self.documents)
        print(f"[DocumentLoader] 总字符数: {total_chars:,}")
        
        return self.documents
    
    def split_documents(self) -> List:
        """
        将文档切分为Chunks (Nodes)
        
        使用SentenceSplitter进行语义感知的切分，
        尽量保持句子完整性，减少信息损失。
        
        Returns:
            List: 切分后的Node列表
        """
        if not self.documents:
            raise ValueError("请先调用load_documents()加载文档")
        
        print(f"[DocumentLoader] 正在切分文档 (chunk_size={self.chunk_size})...")
        
        # 使用SentenceSplitter，比TokenTextSplitter更保留语义
        parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",  # 优先按段落切分
        )
        
        self.nodes = parser.get_nodes_from_documents(self.documents)
        print(f"[DocumentLoader] 切分完成，共 {len(self.nodes)} 个chunks")
        
        # 统计chunk长度分布
        lengths = [len(node.text) for node in self.nodes]
        print(f"[DocumentLoader] Chunk长度 - 平均:{sum(lengths)/len(lengths):.0f} "
              f"最小:{min(lengths)} 最大:{max(lengths)}")
        
        return self.nodes
    
    def build_index(self) -> VectorStoreIndex:
        """
        构建向量索引
        
        使用ChromaDB作为后端（轻量级，本地存储），
        支持后续的相似度检索。
        
        Returns:
            VectorStoreIndex: 构建好的索引
        """
        if not self.nodes:
            raise ValueError("请先调用split_documents()切分文档")
        
        print("[DocumentLoader] 正在构建向量索引...")
        
        # 构建ServiceContext，指定使用轻量级Embedding
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            llm=None,  # 构建索引不需要LLM，节省资源
        )
        
        self.index = VectorStoreIndex(
            nodes=self.nodes,
            service_context=service_context,
            show_progress=True,
        )
        
        print("[DocumentLoader] 索引构建完成")
        return self.index
    
    def save_index(self, save_dir: str = "data/processed/index"):
        """
        保存索引到本地，避免重复构建
        
        Args:
            save_dir: 索引保存目录
        """
        import time
        
        if self.index is None:
            raise ValueError("请先调用build_index()构建索引")
        
        print(f"[DocumentLoader] 正在保存索引到磁盘...")
        start_time = time.time()
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[DocumentLoader] 目标目录: {save_path.absolute()}")
        print(f"[DocumentLoader] 索引包含 {len(self.nodes)} 个chunks，保存中...")
        
        self.index.storage_context.persist(persist_dir=str(save_path))
        
        elapsed = time.time() - start_time
        print(f"[DocumentLoader] 索引已保存到 {save_dir} (耗时 {elapsed:.2f} 秒)")
        
        # 显示保存的文件大小
        try:
            import os
            total_size = sum(os.path.getsize(f) for f in save_path.rglob('*') if f.is_file())
            print(f"[DocumentLoader] 索引文件总大小: {total_size / 1024 / 1024:.2f} MB")
        except:
            pass
    
    def load_index(self, load_dir: str = "data/processed/index") -> VectorStoreIndex:
        """
        从本地加载预构建的索引
        
        Args:
            load_dir: 索引加载目录
            
        Returns:
            VectorStoreIndex: 加载的索引
        """
        from llama_index import StorageContext, load_index_from_storage
        from llama_index.schema import TextNode
        
        load_path = Path(load_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"索引目录不存在: {load_dir}")
        
        print(f"[DocumentLoader] 正在从 {load_dir} 加载索引...")
        
        storage_context = StorageContext.from_defaults(persist_dir=str(load_path))
        
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            llm=None,
        )
        
        self.index = load_index_from_storage(
            storage_context,
            service_context=service_context,
        )
        
        # 从索引中提取 nodes
        try:
            # 尝试从索引中获取文档存储的节点
            docstore = storage_context.docstore
            node_ids = list(docstore.docs.keys())
            self.nodes = [docstore.get_node(node_id) for node_id in node_ids]
            print(f"[DocumentLoader] 从索引中提取了 {len(self.nodes)} 个nodes")
        except Exception as e:
            print(f"[Warning] 无法从索引提取nodes: {e}")
            print("[Warning] 将使用空列表，可能需要重新构建索引")
            self.nodes = []
        
        print("[DocumentLoader] 索引加载完成")
        return self.index
    
    def get_retriever(self, similarity_top_k: int = 5):
        """
        获取索引检索器
        
        Args:
            similarity_top_k: 返回最相似的k个结果
            
        Returns:
            BaseRetriever: 检索器实例
        """
        if self.index is None:
            raise ValueError("请先构建或加载索引")
        
        return self.index.as_retriever(similarity_top_k=similarity_top_k)


if __name__ == "__main__":
    # 使用示例（供测试）
    loader = DocumentLoader(
        data_dir="data/raw",
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # 加载文档
    docs = loader.load_documents(required_exts=[".md", ".txt"])
    
    # 切分并构建索引
    loader.split_documents()
    loader.build_index()
    loader.save_index()
