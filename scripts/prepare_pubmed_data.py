#!/usr/bin/env python3
"""
PubMed RCT 数据预处理脚本

功能：
1. 解析PubMed RCT原始txt文件
2. 转换为Markdown格式（供LlamaIndex使用）
3. 生成适合数据合成的chunks

使用方法：
    python scripts/prepare_pubmed_data.py --input data/raw/pubmed-rct-master/PubMed_20k_RCT/train.txt --output data/processed/pubmed

或者转换所有文件：
    python scripts/prepare_pubmed_data.py --all --input data/raw/pubmed-rct-master --output data/processed/pubmed
"""

import sys
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 添加src到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_synthesis import PubMedRCTParser


def main():
    parser = argparse.ArgumentParser(
        description="PubMed RCT 数据预处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python scripts/prepare_pubmed_data.py \\
      --input data/raw/pubmed-rct-master/PubMed_20k_RCT/train.txt \\
      --output data/processed/pubmed/train

  # 转换所有文件（推荐）
  python scripts/prepare_pubmed_data.py --all \\
      --input data/raw/pubmed-rct-master \\
      --output data/processed/pubmed
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件或目录路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换目录下的所有txt文件（train/dev/test）"
    )
    
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="最大处理文章数（用于快速测试）"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.all:
        # 批量转换模式
        if not input_path.is_dir():
            print(f"[错误] --all模式下输入必须是目录: {input_path}")
            return
        
        # 寻找所有txt文件
        txt_files = list(input_path.rglob("*.txt"))
        
        if not txt_files:
            print(f"[错误] 未找到任何.txt文件: {input_path}")
            return
        
        print(f"[批量转换] 发现 {len(txt_files)} 个txt文件")
        
        # 包装文件处理进度条
        file_iterator = txt_files
        if TQDM_AVAILABLE:
            file_iterator = tqdm(txt_files, desc="处理文件", unit="个")
        
        for txt_file in file_iterator:
            # 确定输出子目录
            # 例如: PubMed_20k_RCT/train.txt -> pubmed_20k/train/
            parent_name = txt_file.parent.name.lower()  # pubmed_20k_rct
            file_stem = txt_file.stem  # train
            
            sub_output = output_path / parent_name.replace("_numbers_replaced_with_at_sign", "") / file_stem
            
            if not TQDM_AVAILABLE:
                print(f"\n[处理] {txt_file}")
            
            try:
                parser = PubMedRCTParser(str(txt_file))
                parser.parse(show_progress=TQDM_AVAILABLE)
                parser.to_markdown_files(str(sub_output), max_files=args.max_articles, show_progress=TQDM_AVAILABLE)
            except Exception as e:
                print(f"[错误] 处理失败: {e}")
                continue
        
        print(f"\n[完成] 所有文件已转换到: {output_path}")
    
    else:
        # 单文件转换模式
        if not input_path.is_file():
            print(f"[错误] 输入文件不存在: {input_path}")
            return
        
        print(f"[单文件转换] {input_path} -> {output_path}")
        
        parser = PubMedRCTParser(str(input_path))
        parser.parse(show_progress=True)
        parser.to_markdown_files(str(output_path), max_files=args.max_articles, show_progress=True)
        
        print(f"[完成] 转换完成！")


if __name__ == "__main__":
    main()
