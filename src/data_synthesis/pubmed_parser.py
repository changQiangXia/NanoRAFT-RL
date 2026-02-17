"""
PubMed RCT 数据集专用解析器

数据集格式说明：
每个样本由多行组成，格式如下：

###24293588
BACKGROUND|This study evaluated the efficacy of...
OBJECTIVE|To determine whether...
METHODS|A randomized controlled trial was conducted...
RESULTS|Of 234 participants randomized...
CONCLUSIONS|Among adults with ...

分隔符：###后跟文章ID
标签：BACKGROUND|、OBJECTIVE|、METHODS|、RESULTS|、CONCLUSIONS|
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[Warning] tqdm未安装，将不显示进度条。安装: pip install tqdm")


@dataclass
class PubMedArticle:
    """PubMed文章数据结构"""
    article_id: str
    background: str
    objective: str
    methods: str
    results: str
    conclusions: str
    
    @property
    def full_text(self) -> str:
        """组合完整文本"""
        parts = []
        if self.background:
            parts.append(f"BACKGROUND: {self.background}")
        if self.objective:
            parts.append(f"OBJECTIVE: {self.objective}")
        if self.methods:
            parts.append(f"METHODS: {self.methods}")
        if self.results:
            parts.append(f"RESULTS: {self.results}")
        if self.conclusions:
            parts.append(f"CONCLUSIONS: {self.conclusions}")
        return "\n\n".join(parts)
    
    def get_section(self, section_name: str) -> str:
        """获取特定章节"""
        return getattr(self, section_name.lower(), "")


class PubMedRCTParser:
    """
    PubMed RCT数据集解析器
    
    解析特殊的结构化医学论文摘要格式
    """
    
    SECTION_LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    
    def __init__(self, file_path: str):
        """
        初始化解析器
        
        Args:
            file_path: .txt文件路径（train.txt, dev.txt, 或 test.txt）
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.articles: List[PubMedArticle] = []
    
    def parse(self, show_progress: bool = True) -> List[PubMedArticle]:
        """
        解析整个文件
        
        Args:
            show_progress: 是否显示进度条
            
        Returns:
            List[PubMedArticle]: 文章列表
        """
        print(f"[PubMedParser] 正在解析 {self.file_path}...")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按文章ID分割（###开头），支持不同换行符(\n, \r\n)
        # 使用更宽松的正则：###后跟数字，然后是任意换行符
        raw_articles = re.split(r'###(\d+)\r?\n', content)
        
        # raw_articles[0]是空字符串或文件头，从1开始
        # 格式: ["", "24293588", "内容...", "24293589", "内容...", ...]
        
        # 预估文章数量
        total_articles = (len(raw_articles) - 1) // 2
        print(f"[PubMedParser] 检测到约 {total_articles} 篇文章")
        
        articles = []
        iterator = range(1, len(raw_articles), 2)
        
        # 包装进度条
        if show_progress and TQDM_AVAILABLE and total_articles > 100:
            iterator = tqdm(iterator, total=total_articles, desc="解析文章", unit="篇")
        
        for i in iterator:
            if i + 1 < len(raw_articles):
                article_id = raw_articles[i].strip()
                article_content = raw_articles[i + 1].strip()
                
                # 检查内容是否为空
                if not article_content:
                    continue
                
                article = self._parse_article(article_id, article_content)
                if article:
                    articles.append(article)
        
        self.articles = articles
        print(f"[PubMedParser] 成功解析 {len(articles)} 篇文章")
        
        # 打印统计
        self._print_stats(articles)
        
        return articles
    
    def _parse_article(self, article_id: str, content: str) -> Optional[PubMedArticle]:
        """
        解析单篇文章
        
        数据格式注意：
        - 字段分隔符是制表符(\t)，不是竖线(|)
        - 例如: "OBJECTIVE\tTo investigate..."
        
        Args:
            article_id: 文章ID
            content: 文章内容
            
        Returns:
            PubMedArticle: 解析后的文章对象
        """
        sections = {label.lower(): "" for label in self.SECTION_LABELS}
        
        # 按行解析
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是标签行（支持制表符或空格分隔）
            for label in self.SECTION_LABELS:
                # 尝试制表符分隔
                if line.startswith(f"{label}\t"):
                    text = line[len(label)+1:].strip()
                    sections[label.lower()] = text
                    break
                # 尝试空格分隔（备用）
                elif line.startswith(f"{label} "):
                    text = line[len(label)+1:].strip()
                    sections[label.lower()] = text
                    break
        
        return PubMedArticle(
            article_id=article_id,
            background=sections["background"],
            objective=sections["objective"],
            methods=sections["methods"],
            results=sections["results"],
            conclusions=sections["conclusions"],
        )
    
    def _print_stats(self, articles: List[PubMedArticle]):
        """打印数据集统计"""
        print(f"\n[PubMedParser] 数据集统计:")
        print(f"  - 总文章数: {len(articles)}")
        
        # 各字段非空统计
        for label in self.SECTION_LABELS:
            count = sum(1 for a in articles if getattr(a, label.lower()))
            print(f"  - 含{label}: {count}篇 ({count/len(articles)*100:.1f}%)")
        
        # 平均长度
        avg_len = sum(len(a.full_text) for a in articles) / len(articles)
        print(f"  - 平均长度: {avg_len:.0f}字符")
        print()
    
    def to_markdown_files(self, output_dir: str, max_files: Optional[int] = None, show_progress: bool = True):
        """
        将解析的文章转换为Markdown文件（供LlamaIndex读取）
        
        Args:
            output_dir: 输出目录
            max_files: 最大转换数量（None表示全部）
            show_progress: 是否显示进度条
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        articles_to_convert = self.articles[:max_files] if max_files else self.articles
        
        print(f"[PubMedParser] 正在转换 {len(articles_to_convert)} 篇文章为Markdown...")
        
        # 包装进度条
        article_iterator = articles_to_convert
        if show_progress and TQDM_AVAILABLE and len(articles_to_convert) > 100:
            article_iterator = tqdm(articles_to_convert, desc="生成Markdown", unit="篇")
        
        for article in article_iterator:
            md_content = f"""# PubMed Article {article.article_id}

## BACKGROUND
{article.background}

## OBJECTIVE
{article.objective}

## METHODS
{article.methods}

## RESULTS
{article.results}

## CONCLUSIONS
{article.conclusions}

---
*Source: PubMed RCT Dataset*
"""
            
            output_file = output_path / f"pubmed_{article.article_id}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        print(f"[PubMedParser] Markdown文件已保存到 {output_dir}")
    
    def get_chunks_for_synthesis(self, chunk_type: str = "full") -> List[Dict[str, str]]:
        """
        获取用于数据合成的文本块
        
        Args:
            chunk_type: 
                - "full": 整篇文章
                - "by_section": 按章节分割
                - "methods_only": 仅METHODS章节（适合方法类问题）
                - "conclusions_only": 仅CONCLUSIONS章节
                
        Returns:
            List[Dict]: 包含text和metadata的chunk列表
        """
        chunks = []
        
        for article in self.articles:
            if chunk_type == "full":
                chunks.append({
                    "text": article.full_text,
                    "metadata": {
                        "article_id": article.article_id,
                        "section": "full",
                        "type": "pubmed_rct"
                    }
                })
            
            elif chunk_type == "by_section":
                for section in self.SECTION_LABELS:
                    text = getattr(article, section.lower())
                    if text:
                        chunks.append({
                            "text": f"{section}: {text}",
                            "metadata": {
                                "article_id": article.article_id,
                                "section": section,
                                "type": "pubmed_rct"
                            }
                        })
            
            elif chunk_type == "methods_only":
                if article.methods:
                    chunks.append({
                        "text": article.methods,
                        "metadata": {
                            "article_id": article.article_id,
                            "section": "METHODS",
                            "type": "pubmed_rct"
                        }
                    })
            
            elif chunk_type == "conclusions_only":
                if article.conclusions:
                    chunks.append({
                        "text": article.conclusions,
                        "metadata": {
                            "article_id": article.article_id,
                            "section": "CONCLUSIONS",
                            "type": "pubmed_rct"
                        }
                    })
        
        return chunks


def convert_pubmed_to_format(
    input_file: str,
    output_dir: str,
    format_type: str = "markdown",
    max_articles: Optional[int] = None,
):
    """
    便捷的转换函数
    
    Args:
        input_file: PubMed .txt文件路径
        output_dir: 输出目录
        format_type: 输出格式 ["markdown"]
        max_articles: 最大处理文章数
    """
    parser = PubMedRCTParser(input_file)
    parser.parse()
    
    if format_type == "markdown":
        parser.to_markdown_files(output_dir, max_files=max_articles)
    
    print(f"[Convert] 转换完成: {input_file} -> {output_dir}")


if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pubmed_parser.py <train.txt或dev.txt路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    parser = PubMedRCTParser(file_path)
    articles = parser.parse()
    
    # 显示第一篇示例
    if articles:
        print("\n示例文章:")
        print(f"ID: {articles[0].article_id}")
        print(f"内容预览:\n{articles[0].full_text[:500]}...")
