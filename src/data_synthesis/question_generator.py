"""
问题生成器 - 基于LangChain

功能：
1. 使用教师模型（GPT-4/Claude/智谱AI等API）从chunk中抽取信息
2. 生成需要多步推理的问题（非简单事实抽取）
3. 生成Chain-of-Thought推理过程和答案

输出格式要求：
{
    "question": "需要推理的问题",
    "chain_of_thought": "详细的推理步骤",
    "answer": "最终答案",
    "source_chunk": "来源chunk文本"
}
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 智谱AI支持
try:
    from zhipuai import ZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False


class QAOutput(BaseModel):
    """
    问题-答案输出的结构化定义
    
    使用Pydantic约束LLM输出格式，确保数据质量。
    """
    question: str = Field(
        description="基于文档内容提出的需要推理的问题，不能是简单的事实查找"
    )
    chain_of_thought: str = Field(
        description="详细的推理过程，展示如何从文档中逐步得出答案"
    )
    answer: str = Field(
        description="问题的最终答案，应该简洁但完整"
    )
    reasoning_type: str = Field(
        description="推理类型，如'多步推理'/'对比分析'/'因果推断'等"
    )


@dataclass
class GeneratedQA:
    """
    生成的问题-答案数据类
    
    Attributes:
        question: 问题文本
        chain_of_thought: CoT推理过程
        answer: 答案
        reasoning_type: 推理类型标签
        source_chunk: 来源文档chunk
        metadata: 额外元数据
    """
    question: str
    chain_of_thought: str
    answer: str
    reasoning_type: str
    source_chunk: str
    metadata: Dict[str, Any]


class QuestionGenerator:
    """
    问题生成器
    
    支持OpenAI兼容API和智谱AI API。
    
    Attributes:
        provider: API提供商 ["openai", "zhipu"]
        llm: LangChain LLM实例 (OpenAI模式)
        zhipu_client: 智谱AI客户端 (智谱AI模式)
    """
    
    # 系统提示词 - 定义教师模型的角色和任务
    SYSTEM_PROMPT = """你是专业的医学研究分析专家。你的任务是基于给定的医学文献片段（来自PubMed RCT数据集），生成高质量的问答对。

核心要求：
1. **问题设计**：必须是需要推理的问题，不能是简单的事实查找。例如：
   - ❌ 差："这项研究使用了多少参与者？"（直接查找）
   - ✅ 好："考虑到这项研究的样本量和试验设计，为什么作者选择使用低剂量泼尼松而非安慰剂作为对照，这个剂量选择如何影响研究结论的可靠性？"（需要推理）

2. **推理过程**：必须展示完整的Chain-of-Thought推理，包括：
   - 识别关键信息
   - 分析约束条件
   - 逐步推导
   - 得出结论

3. **领域聚焦**：问题必须围绕医学RCT研究、临床试验设计、疗效评估等专业主题。

4. **输出格式**：必须严格按照JSON格式输出，包含question、chain_of_thought、answer、reasoning_type四个字段。
"""

    # 人类提示词模板
    HUMAN_PROMPT_TEMPLATE = """请基于以下文档片段生成一个问答对：

[文档片段]
{chunk_text}

[要求]
- 问题必须需要2-3步推理才能回答
- 推理过程要详细展示思考链条
- 答案要准确且完整
- 如果文档是医学RCT研究，问题应该涉及研究设计、疗效机制或临床意义

请直接输出JSON格式：
{{
    "question": "你的问题",
    "chain_of_thought": "逐步推理过程...",
    "answer": "最终答案",
    "reasoning_type": "推理类型标签"
}}
"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        provider: str = "openai",  # 新增：提供商 ["openai", "zhipu"]
    ):
        """
        初始化问题生成器
        
        Args:
            model_name: 使用的模型名称
            api_key: API Key（默认从环境变量读取）
            base_url: API基础URL（OpenAI兼容格式）
            temperature: 生成温度（0-1，越高越创造性）
            max_retries: 失败重试次数
            provider: API提供商 ["openai", "zhipu"]
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        
        if self.provider == "zhipu":
            self._init_zhipu(api_key, temperature)
        else:
            self._init_openai(api_key, base_url, temperature, max_retries)
    
    def _init_openai(self, api_key, base_url, temperature, max_retries):
        """初始化OpenAI兼容API"""
        # 从环境变量读取API Key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("[Warning] 未设置OPENAI_API_KEY，请设置环境变量或在初始化时传入")
        
        llm_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_retries": max_retries,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # 设置输出解析器
        self.parser = JsonOutputParser(pydantic_object=QAOutput)
        
        # 构建提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(self.HUMAN_PROMPT_TEMPLATE),
        ])
        
        # 构建处理链
        self.chain = self.prompt | self.llm | self.parser
        
        print(f"[QuestionGenerator] 初始化完成(OpenAI)，使用模型: {self.model_name}")
    
    def _init_zhipu(self, api_key, temperature):
        """初始化智谱AI API"""
        print(f"[QuestionGenerator] 开始初始化智谱AI...")
        
        if not ZHIPU_AVAILABLE:
            raise ImportError("智谱AI支持需要安装zhipuai: pip install zhipuai")
        
        # 从环境变量或参数读取API Key
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            raise ValueError("使用智谱AI需要提供api_key或设置ZHIPU_API_KEY环境变量")
        
        print(f"[QuestionGenerator] 正在连接智谱AI服务...")
        self.zhipu_client = ZhipuAI(api_key=self.api_key)
        self.zhipu_temperature = temperature
        
        print(f"[QuestionGenerator] 初始化完成(智谱AI)，使用模型: {self.model_name}")
    
    def _parse_zhipu_response(self, response_text: str) -> Optional[Dict]:
        """
        解析智谱AI的响应，提取JSON
        
        Args:
            response_text: API返回的文本
            
        Returns:
            Dict: 解析后的JSON，失败返回None
        """
        try:
            # 尝试直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            # 查找花括号包裹的内容
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            
            print(f"[Error] 无法解析智谱AI响应: {response_text[:200]}")
            return None
    
    def generate_from_chunk(
        self,
        chunk_text: str,
        max_length: int = 2000,
    ) -> Optional[GeneratedQA]:
        """
        从单个chunk生成问答对
        
        Args:
            chunk_text: 文档chunk文本
            max_length: 最大处理长度（超长会截断）
            
        Returns:
            GeneratedQA: 生成的问答对，失败返回None
        """
        # 截断过长的chunk
        if len(chunk_text) > max_length:
            chunk_text = chunk_text[:max_length] + "..."
        
        try:
            if self.provider == "zhipu":
                return self._generate_from_chunk_zhipu(chunk_text)
            else:
                return self._generate_from_chunk_openai(chunk_text)
            
        except Exception as e:
            print(f"[Error] 生成失败: {e}")
            return None
    
    def _generate_from_chunk_openai(self, chunk_text: str) -> Optional[GeneratedQA]:
        """使用OpenAI兼容API生成"""
        result = self.chain.invoke({"chunk_text": chunk_text})
        
        return GeneratedQA(
            question=result["question"],
            chain_of_thought=result["chain_of_thought"],
            answer=result["answer"],
            reasoning_type=result.get("reasoning_type", "general"),
            source_chunk=chunk_text,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "provider": "openai",
            }
        )
    
    def _generate_from_chunk_zhipu(self, chunk_text: str) -> Optional[GeneratedQA]:
        """使用智谱AI生成"""
        # 构建完整提示词
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{self.HUMAN_PROMPT_TEMPLATE.format(chunk_text=chunk_text)}"
        
        # 调用智谱AI API
        response = self.zhipu_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"请基于以下文档片段生成一个问答对：\n\n[文档片段]\n{chunk_text}\n\n[要求]\n- 问题必须需要2-3步推理才能回答\n- 推理过程要详细展示思考链条\n- 答案要准确且完整\n- 如果文档是医学RCT研究，问题应该涉及研究设计、疗效机制或临床意义\n\n请直接输出JSON格式。"}
            ],
            temperature=self.zhipu_temperature,
        )
        
        # 解析响应
        response_text = response.choices[0].message.content
        result = self._parse_zhipu_response(response_text)
        
        if not result:
            return None
        
        return GeneratedQA(
            question=result.get("question", ""),
            chain_of_thought=result.get("chain_of_thought", ""),
            answer=result.get("answer", ""),
            reasoning_type=result.get("reasoning_type", "general"),
            source_chunk=chunk_text,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "provider": "zhipu",
            }
        )
    
    def generate_batch(
        self,
        chunks: List[str],
        max_length: int = 2000,
        verbose: bool = True,
    ) -> List[GeneratedQA]:
        """
        批量生成问答对
        
        Args:
            chunks: chunk文本列表
            max_length: 最大处理长度
            verbose: 是否打印进度
            
        Returns:
            List[GeneratedQA]: 生成的问答对列表
        """
        results = []
        
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"[QuestionGenerator] 处理 {i+1}/{len(chunks)}...", end="\r")
            
            qa = self.generate_from_chunk(chunk, max_length)
            if qa:
                results.append(qa)
        
        if verbose:
            print(f"\n[QuestionGenerator] 成功生成 {len(results)}/{len(chunks)} 个问答对")
        
        return results
    
    def generate_with_distractors(
        self,
        chunk_text: str,
        distractor_chunks: List[str],
    ) -> Optional[GeneratedQA]:
        """
        在干扰项存在的情况下生成问题（更高难度）
        
        Args:
            chunk_text: 正确的文档chunk
            distractor_chunks: 干扰项chunks
            
        Returns:
            GeneratedQA: 生成的问答对
        """
        # 构建特殊提示词
        context_parts = ["[核心文档]\n" + chunk_text]
        for i, dist in enumerate(distractor_chunks, 1):
            context_parts.append(f"[相关文档{i}]\n{dist[:500]}...")
        
        full_context = "\n\n".join(context_parts)
        
        special_system_prompt = self.SYSTEM_PROMPT + "\n\n注意：文档中包含核心信息和相关参考信息，问题必须能够通过核心文档准确回答。"
        
        try:
            if self.provider == "zhipu":
                # 智谱AI模式
                response = self.zhipu_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": special_system_prompt},
                        {"role": "user", "content": f"请基于以下多份文档片段生成一个问答对：\n\n{full_context}\n\n[要求]\n- 问题必须主要依赖[核心文档]中的信息回答\n- 相关参考文档可能包含干扰信息\n- 展示如何从核心文档中提取关键信息\n\n请直接输出JSON格式。"}
                    ],
                    temperature=self.zhipu_temperature,
                )
                response_text = response.choices[0].message.content
                result = self._parse_zhipu_response(response_text)
            else:
                # OpenAI模式
                special_prompt = special_system_prompt + "\n\n请基于以下多份文档片段生成一个问答对：\n\n{context}\n\n[要求]\n- 问题必须主要依赖[核心文档]中的信息回答\n- 相关参考文档可能包含干扰信息\n- 展示如何从核心文档中提取关键信息\n\n请直接输出JSON格式。"
                
                messages = [
                    ("system", special_system_prompt),
                    ("human", special_prompt.format(context=full_context)),
                ]
                result = self.llm.invoke(messages)
                result = self.parser.parse(result.content)
            
            if not result:
                return None
            
            return GeneratedQA(
                question=result.get("question", ""),
                chain_of_thought=result.get("chain_of_thought", ""),
                answer=result.get("answer", ""),
                reasoning_type=result.get("reasoning_type", "with_distractors"),
                source_chunk=chunk_text,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "provider": self.provider,
                    "has_distractors": True,
                    "num_distractors": len(distractor_chunks),
                }
            )
            
        except Exception as e:
            print(f"[Error] 带干扰项生成失败: {e}")
            return None


class LocalModelGenerator(QuestionGenerator):
    """
    本地模型问题生成器（备选方案）
    
    当API不可用时，可使用本地小模型生成问答对。
    质量可能略低，但无需API费用。
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/phi-2",
        device: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu",
    ):
        """
        使用本地模型初始化
        
        Args:
            model_path: 本地模型路径或HuggingFace模型名
            device: 运行设备
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        print(f"[LocalModelGenerator] 加载本地模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "cpu":
            self.model = self.model.to("cpu")
        
        print("[LocalModelGenerator] 模型加载完成")
    
    def generate_from_chunk(self, chunk_text: str, max_length: int = 2000) -> Optional[GeneratedQA]:
        """使用本地模型生成（简化版）"""
        # 构建prompt
        prompt = f"""Based on the following text, generate a question and answer:

Text: {chunk_text[:1500]}

Question: """
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 简单解析（本地模型输出格式可能不稳定）
        try:
            parts = generated.split("Answer:")
            question = parts[0].split("Question:")[-1].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            
            return GeneratedQA(
                question=question,
                chain_of_thought="Generated by local model",
                answer=answer,
                reasoning_type="local_model",
                source_chunk=chunk_text,
                metadata={"model": "local"}
            )
        except:
            return None


if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python question_generator.py <provider> [api_key]")
        print("  provider: openai 或 zhipu")
        sys.exit(1)
    
    provider = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if provider == "zhipu":
        generator = QuestionGenerator(
            model_name="glm-4",
            provider="zhipu",
            api_key=api_key,
        )
    else:
        generator = QuestionGenerator(model_name="gpt-3.5-turbo")
    
    sample_chunk = """
    Selenium WebDriver provides a way to locate elements within Shadow DOM using 
    JavaScript execution. When an element is inside Shadow DOM, traditional 
    find_element methods fail because Shadow DOM creates a separate scope. 
    The solution is to use execute_script to pierce through the Shadow boundary.
    """
    
    qa = generator.generate_from_chunk(sample_chunk)
    if qa:
        print(f"问题: {qa.question}")
        print(f"推理: {qa.chain_of_thought}")
        print(f"答案: {qa.answer}")
