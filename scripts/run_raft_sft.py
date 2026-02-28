#!/usr/bin/env python3
"""
RAFT (Retrieval-Augmented Fine-Tuning) SFT 训练脚本

使用方法:
    python scripts/run_raft_sft.py [--config configs/raft_sft.yaml]

功能:
1. 4-bit量化加载模型
2. LoRA微调
3. 针对4GB显存优化
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# 设置镜像（避免HuggingFace下载超时）
# 允许外部通过环境变量覆盖（例如HF_ENDPOINT=https://huggingface.co）
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 修复Accelerate版本兼容性
os.environ["ACCELERATE_USE_SEEDABLE_SAMPLER"] = "false"

# 设置模型缓存目录为项目内目录（避免占用C盘空间）
PROJECT_ROOT = Path(__file__).parent.parent
os.environ["HF_HOME"] = str(PROJECT_ROOT / "models" / "cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(PROJECT_ROOT / "models" / "cache")
print(f"[Setup] 模型缓存目录: {os.environ['HF_HOME']}")

# 添加src到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.raft.model_loader import load_model_and_tokenizer, print_gpu_memory
from src.raft.dataset import create_dataset, tokenize_dataset
from src.raft.trainer import train, print_training_summary
from src.utils.logging_utils import setup_logger


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/raft_sft.yaml"):
    """
    RAFT SFT主流程
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    print(f"[Main] 加载配置: {config_path}")
    
    # 设置日志
    logger = setup_logger(
        name="raft_sft",
        level="INFO",
        log_dir="logs/raft_sft",
    )
    logger.info("=" * 60)
    logger.info("RAFT SFT训练启动")
    logger.info(f"时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # 打印配置摘要
    print_training_summary(config)
    
    # ============================================================
    # 步骤1: 加载模型和分词器
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤1/4: 加载模型和分词器")
    print("=" * 60)
    
    print_gpu_memory()
    model, tokenizer = load_model_and_tokenizer(config)
    print_gpu_memory()
    
    # ============================================================
    # 步骤2: 准备数据集
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤2/4: 准备数据集")
    print("=" * 60)
    
    data_config = config["data"]
    raft_config = config.get("raft", {})
    response_template = raft_config.get("response_template", "### Response:\n")
    
    # 加载训练集
    train_dataset = create_dataset(data_config["train_file"], response_template)
    train_dataset = tokenize_dataset(
        train_dataset, 
        tokenizer, 
        data_config["max_seq_length"]
    )
    
    # 加载验证集（如果存在）
    eval_dataset = None
    if Path(data_config["val_file"]).exists():
        eval_dataset = create_dataset(data_config["val_file"], response_template)
        eval_dataset = tokenize_dataset(
            eval_dataset,
            tokenizer,
            data_config["max_seq_length"]
        )
    
    # ============================================================
    # 步骤3: 训练
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤3/4: 开始训练")
    print("=" * 60)
    
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        train_config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # ============================================================
    # 步骤4: 保存和总结
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤4/4: 训练完成")
    print("=" * 60)
    
    output_dir = config["training"]["output_dir"]
    final_model_dir = Path(output_dir) / "final"
    
    print(f"✅ 训练完成！")
    print(f"📁 模型保存位置: {final_model_dir}")
    print(f"🎯 下一步: 运行PPO强化学习或模型评估")
    print()
    print("使用模型进行推理:")
    print(f"  from peft import AutoPeftModelForCausalLM")
    print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{final_model_dir}')")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAFT SFT训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置训练
  python scripts/run_raft_sft.py
  
  # 使用自定义配置
  python scripts/run_raft_sft.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/raft_sft.yaml",
        help="配置文件路径 (默认: configs/raft_sft.yaml)"
    )
    
    args = parser.parse_args()
    
    main(args.config)
