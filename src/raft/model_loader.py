"""
RAFT QLoRA 模型加载器

功能：
1. 4-bit量化加载大模型
2. 配置LoRA适配器
3. 针对4GB显存极致优化
"""

import sys
import torch

# 在Windows上阻止bitsandbytes加载（有兼容性问题）
if sys.platform == "win32":
    sys.modules['bitsandbytes'] = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def load_model_and_tokenizer(config: dict):
    """
    加载QLoRA模型和分词器
    
    Args:
        config: 配置字典
        
    Returns:
        model: 配置好的PEFT模型
        tokenizer: 分词器
    """
    model_config = config["model"]
    qlora_config = config["qlora"]
    lora_config = config["lora"]
    
    model_name = model_config["base_model"]
    
    print(f"[ModelLoader] 正在加载模型: {model_name}")
    print(f"[ModelLoader] 使用4-bit量化: {qlora_config['enabled']}")
    
    # ============================================================
    # 1. 配置4-bit量化
    # ============================================================
    if qlora_config["enabled"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config["load_in_4bit"],
            bnb_4bit_quant_type=qlora_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_config["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=qlora_config["bnb_4bit_use_double_quant"],
        )
    else:
        bnb_config = None
    
    # ============================================================
    # 2. 加载模型
    # ============================================================
    # 清理显存
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                    # 自动分配层到GPU/CPU
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
        # 限制显存使用
        max_memory={0: "22GiB", "cpu": "50GiB"} if torch.cuda.is_available() else None,  # 4090可用22GB
    )
    
    print(f"[ModelLoader] 模型加载完成")
    print(f"[ModelLoader] 设备映射: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'default'}")
    
    # ============================================================
    # 3. 加载分词器
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
        padding_side=config["data"].get("padding_side", "right"),
    )
    
    # 设置填充token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 调整模型嵌入层（某些模型需要）
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"[ModelLoader] 分词器加载完成")
    print(f"[ModelLoader] 词汇表大小: {len(tokenizer)}")
    
    # ============================================================
    # 4. 准备模型用于训练（梯度检查点等）
    # ============================================================
    if qlora_config["enabled"]:
        print("[ModelLoader] 准备模型用于4-bit训练...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )
    
    # ============================================================
    # 5. 配置LoRA
    # ============================================================
    print(f"[ModelLoader] 配置LoRA: r={lora_config['r']}, alpha={lora_config['alpha']}")
    
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )
    
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    print("[ModelLoader] 模型准备完成!")
    
    return model, tokenizer


def print_gpu_memory():
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU Memory] 已分配: {allocated:.2f} GB | 预留: {reserved:.2f} GB | 总计: {total:.2f} GB")
    else:
        print("[GPU Memory] 无GPU可用")


if __name__ == "__main__":
    # 测试
    test_config = {
        "model": {"base_model": "Qwen/Qwen2-0.5B-Instruct", "trust_remote_code": True},
        "qlora": {
            "enabled": True,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "lora": {
            "r": 8,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "data": {"padding_side": "right"},
        "training": {"gradient_checkpointing": True},
    }
    
    print("=" * 60)
    print("测试模型加载")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(test_config)
    print_gpu_memory()
