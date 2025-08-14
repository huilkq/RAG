"""
LLM服务

集成OpenAI API和LangSmith，提供智能问答能力
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from loguru import logger

from ..core.config import get_settings
from ..core.logging import log_llm_call
from ..models.document import DocumentChunk
from ..models.chat import SourceReference

settings = get_settings()


class LLMService:
    """LLM服务主类"""
    
    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)
        self.llm = None
        self.tracer = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化LLM模型"""
        try:
            # 配置LangSmith追踪
            if settings.langchain_tracing_v2 and settings.langchain_api_key:
                self.tracer = LangChainTracer(
                    project_name=settings.langchain_project,
                    api_key=settings.langchain_api_key,
                    endpoint=settings.langchain_endpoint
                )
                callback_manager = CallbackManager([self.tracer])
                self.logger.info("LangSmith追踪已启用")
            else:
                callback_manager = None
                self.logger.info("LangSmith追踪未启用")
            
            # 初始化OpenAI模型
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url,
                callbacks=callback_manager
            )
            
            self.logger.info(f"LLM服务初始化完成，模型: {settings.openai_model}")
            
        except Exception as e:
            self.logger.error(f"初始化LLM服务失败: {e}")
            raise RuntimeError(f"LLM服务初始化失败: {e}")
    
    def generate_answer(
        self, 
        question: str, 
        context_chunks: List[DocumentChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[SourceReference], Dict[str, Any]]:
        """生成回答"""
        try:
            start_time = time.time()
            
            # 构建系统提示词
            system_prompt = self._build_system_prompt()
            
            # 构建上下文提示词
            context_prompt = self._build_context_prompt(context_chunks)
            
            # 构建对话历史提示词
            history_prompt = self._build_history_prompt(conversation_history) if conversation_history else ""
            
            # 构建完整提示词
            full_prompt = f"{system_prompt}\n\n{context_prompt}\n\n{history_prompt}\n\n用户问题: {question}\n\n请基于提供的上下文信息回答用户问题。如果上下文中没有相关信息，请明确说明。在回答中请引用相关的文档片段。"
            
            # 调用LLM
            messages = [HumanMessage(content=full_prompt)]
            
            # 设置模型参数
            model_params = {}
            if temperature is not None:
                model_params["temperature"] = temperature
            if max_tokens is not None:
                model_params["max_tokens"] = max_tokens
            
            response = self.llm.invoke(messages, **model_params)
            answer = response.content
            
            # 提取引用来源
            sources = self._extract_sources(context_chunks)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 记录性能日志
            log_llm_call(
                model=settings.openai_model,
                prompt_tokens=len(full_prompt.split()),
                completion_tokens=len(answer.split()),
                total_time=processing_time
            )
            
            # 构建返回的元数据
            metadata = {
                "model": settings.openai_model,
                "temperature": temperature or settings.temperature,
                "max_tokens": max_tokens or settings.max_tokens,
                "context_chunks_count": len(context_chunks),
                "processing_time": processing_time
            }
            
            self.logger.info(f"LLM回答生成完成，问题长度: {len(question)}, 回答长度: {len(answer)}, 耗时: {processing_time:.3f}s")
            
            return answer, sources, metadata
            
        except Exception as e:
            self.logger.error(f"生成LLM回答失败: {e}")
            raise RuntimeError(f"LLM回答生成失败: {e}")
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的智能问答助手，专门基于提供的文档内容回答用户问题。

你的主要职责：
1. 仔细阅读和理解提供的文档上下文
2. 基于上下文信息准确回答用户问题
3. 如果上下文中没有相关信息，明确说明
4. 在回答中引用相关的文档片段
5. 保持回答的准确性和相关性
6. 使用清晰、易懂的语言

回答要求：
- 必须基于提供的上下文信息
- 引用具体的文档片段
- 如果信息不足，明确说明
- 保持客观、专业的语调"""
    
    def _build_context_prompt(self, context_chunks: List[DocumentChunk]) -> str:
        """构建上下文提示词"""
        if not context_chunks:
            return "上下文信息：无相关文档内容。"
        
        context_parts = ["上下文信息："]
        
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"文档片段 {i}:")
            context_parts.append(f"内容: {chunk.content}")
            context_parts.append(f"来源: 文档ID {chunk.document_id}, 片段 {chunk.chunk_index}")
            context_parts.append("")  # 空行分隔
        
        return "\n".join(context_parts)
    
    def _build_history_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """构建对话历史提示词"""
        if not conversation_history:
            return ""
        
        history_parts = ["对话历史："]
        
        for i, msg in enumerate(conversation_history[-3:], 1):  # 只保留最近3轮对话
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def _extract_sources(self, context_chunks: List[DocumentChunk]) -> List[SourceReference]:
        """提取引用来源"""
        sources = []
        
        for chunk in context_chunks:
            source = SourceReference(
                document_id=chunk.document_id,
                document_title=f"文档 {chunk.document_id}",  # 这里可以扩展为实际的文档标题
                chunk_id=chunk.id,
                content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                similarity_score=1.0,  # 这里可以传入实际的相似度分数
                start_char=chunk.start_char,
                end_char=chunk.end_char
            )
            sources.append(source)
        
        return sources
    
    def generate_conversation_response(
        self, 
        message: str, 
        conversation_history: List[Dict[str, str]],
        context_chunks: Optional[List[DocumentChunk]] = None
    ) -> Tuple[str, List[SourceReference], Dict[str, Any]]:
        """生成对话响应"""
        try:
            # 如果没有提供上下文，尝试从对话历史中提取相关信息
            if not context_chunks:
                # 这里可以实现基于对话历史的上下文提取逻辑
                # 暂时使用空列表
                context_chunks = []
            
            return self.generate_answer(
                question=message,
                context_chunks=context_chunks,
                conversation_history=conversation_history
            )
            
        except Exception as e:
            self.logger.error(f"生成对话响应失败: {e}")
            raise RuntimeError(f"对话响应生成失败: {e}")
    
    def summarize_conversation(
        self, 
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """总结对话内容"""
        try:
            if not conversation_history:
                return "无对话内容"
            
            # 构建总结提示词
            summary_prompt = f"""请总结以下对话的主要内容：

对话内容：
{chr(10).join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in conversation_history])}

请提供一个简洁的总结，包括：
1. 主要讨论的话题
2. 关键信息点
3. 达成的结论或结果

总结："""
            
            # 调用LLM生成总结
            messages = [HumanMessage(content=summary_prompt)]
            response = self.llm.invoke(messages)
            summary = response.content
            
            self.logger.info(f"对话总结生成完成，对话轮数: {len(conversation_history)}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成对话总结失败: {e}")
            return "生成总结失败"
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取文本关键词"""
        try:
            extract_prompt = f"""请从以下文本中提取{max_keywords}个最重要的关键词：

文本内容：
{text}

请只返回关键词列表，每行一个关键词，不要包含其他内容。"""
            
            messages = [HumanMessage(content=extract_prompt)]
            response = self.llm.invoke(messages)
            
            # 解析关键词
            keywords = [kw.strip() for kw in response.content.split('\n') if kw.strip()]
            
            self.logger.info(f"关键词提取完成，提取数量: {len(keywords)}")
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.error(f"提取关键词失败: {e}")
            return []
    
    def classify_document(self, content: str, categories: List[str]) -> str:
        """文档分类"""
        try:
            categories_str = ", ".join(categories)
            classify_prompt = f"""请将以下文档内容分类到给定的类别中：

文档内容：
{content[:1000]}...

可选类别：{categories_str}

请只返回最合适的类别名称，不要包含其他内容。"""
            
            messages = [HumanMessage(content=classify_prompt)]
            response = self.llm.invoke(messages)
            
            category = response.content.strip()
            
            # 验证分类结果
            if category in categories:
                self.logger.info(f"文档分类完成，类别: {category}")
                return category
            else:
                self.logger.warning(f"文档分类结果不在预定义类别中: {category}")
                return categories[0]  # 返回默认类别
            
        except Exception as e:
            self.logger.error(f"文档分类失败: {e}")
            return categories[0] if categories else "未分类"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": settings.openai_model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "langsmith_enabled": bool(settings.langchain_tracing_v2 and settings.langchain_api_key),
            "langsmith_project": settings.langchain_project
        }
    
    def update_model_parameters(
        self, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> bool:
        """更新模型参数"""
        try:
            if temperature is not None:
                self.llm.temperature = temperature
                self.logger.info(f"模型温度参数已更新: {temperature}")
            
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
                self.logger.info(f"模型最大token参数已更新: {max_tokens}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新模型参数失败: {e}")
            return False
