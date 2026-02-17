#!/usr/bin/env python3
"""
预下载模型脚本

功能：
1. 自动设置Hugging Face镜像
2. 预下载Embedding模型（避免运行时下载超时）
3. 支持ModelScope备用下载

使用方法：
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def download_from_huggingface(model_name: str, cache_dir: str = "./models"):
    """从Hugging Face下载模型（使用镜像）"""
    print(f"[Download] 正在从Hugging Face下载: {model_name}")
    print(f"[Download] 使用镜像: {os.getenv('HF_ENDPOINT')}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 下载模型
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print(f"[Download] ✓ 成功下载: {model_name}")
        return True
        
    except Exception as e:
        print(f"[Download] ✗ 下载失败: {e}")
        return False


def download_from_modelscope(model_name: str, cache_dir: str = "./models"):
    """从ModelScope（魔搭社区）下载模型"""
    print(f"[Download] 尝试从ModelScope下载: {model_name}")
    
    try:
        # ModelScope模型名映射
        modelscope_mapping = {
            "sentence-transformers/all-MiniLM-L6-v2": "AI-ModelScope/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2": "AI-ModelScope/all-MiniLM-L12-v2",
            "BAAI/bge-small-en": "BAAI/bge-small-en",
            "BAAI/bge-large-en": "BAAI/bge-large-en-v1.5",
        }
        
        modelscope_name = modelscope_mapping.get(model_name, model_name)
        
        # 安装modelscope
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("[Download] 正在安装modelscope...")
            os.system(f"{sys.executable} -m pip install modelscope -q")
            from modelscope import snapshot_download
        
        # 下载模型
        local_path = snapshot_download(modelscope_name, cache_dir=cache_dir)
        print(f"[Download] ✓ 成功下载到: {local_path}")
        return True
        
    except Exception as e:
        print(f"[Download] ✗ ModelScope下载失败: {e}")
        return False


def main():
    """主函数：下载所需模型"""
    print("=" * 60)
    print("模型预下载脚本")
    print("=" * 60)
    print()
    
    # 需要下载的模型
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 默认Embedding模型
        # 可以添加更多模型
    ]
    
    cache_dir = "./models"
    Path(cache_dir).mkdir(exist_ok=True)
    
    success_count = 0
    
    for model_name in models:
        print(f"\n[{success_count+1}/{len(models)}] {model_name}")
        print("-" * 40)
        
        # 先尝试Hugging Face镜像
        if download_from_huggingface(model_name, cache_dir):
            success_count += 1
            continue
        
        # 失败则尝试ModelScope
        print("[Download] 尝试备用下载源...")
        if download_from_modelscope(model_name, cache_dir):
            success_count += 1
            continue
        
        print(f"[Download] 所有下载源均失败: {model_name}")
    
    print()
    print("=" * 60)
    print(f"下载完成: {success_count}/{len(models)} 个模型")
    print("=" * 60)
    
    if success_count < len(models):
        print("\n[提示] 部分模型下载失败，您可以:")
        print("  1. 检查网络连接")
        print("  2. 手动下载模型并放入 ./models 目录")
        print("  3. 使用其他模型名称")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
