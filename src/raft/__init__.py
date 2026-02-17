"""
RAFT (Retrieval-Augmented Fine-Tuning) 模块

基于QLoRA的4-bit量化微调，使小模型学会在含噪声的上下文中推理。
"""

from .model_loader import load_model_and_tokenizer, print_gpu_memory
from .dataset import create_dataset, tokenize_dataset
from .trainer import train, print_training_summary

__all__ = [
    "load_model_and_tokenizer",
    "print_gpu_memory",
    "create_dataset",
    "tokenize_dataset",
    "train",
    "print_training_summary",
]
