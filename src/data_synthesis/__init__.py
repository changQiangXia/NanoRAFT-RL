"""
数据合成流水线模块

本模块负责：
1. 使用LlamaIndex加载并切分本地文档
2. 利用LangChain编排教师模型生成问题
3. 注入干扰项(Distractors)构建抗噪训练数据
4. 输出Alpaca格式数据（含Chain-of-Thought）

数据格式：
{
    "instruction": "问题描述",
    "input": "上下文（含干扰项）",
    "output": "推理过程 + 答案"
}
"""

from .document_loader import DocumentLoader
from .retriever import DistractorRetriever
from .question_generator import QuestionGenerator
from .distractor_injector import DistractorInjector
from .alpaca_formatter import AlpacaFormatter
from .pubmed_parser import PubMedRCTParser, convert_pubmed_to_format

__all__ = [
    "DocumentLoader",
    "DistractorRetriever", 
    "QuestionGenerator",
    "DistractorInjector",
    "AlpacaFormatter",
    "PubMedRCTParser",
    "convert_pubmed_to_format",
]
