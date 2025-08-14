"""
LangGraph Agent服务

使用LangGraph构建智能问答的执行流程图
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from loguru import logger

from ..core.config import get_settings
from ..core.logging import log_performance
from ..models.document import DocumentChunk
from ..models.chat import SourceReference, Message, Conversation
from .vector_service import VectorService
from .llm_service import LLMService

settings = get_settings()


class AgentState:
    """Agent状态类"""
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.current_question: str = ""
        self.context_chunks: List[DocumentChunk] = []
        self.sources: List[SourceReference] = []
        self.answer: str = ""
        self.conversation_id: str = ""
        self.metadata: Dict[str, Any] = {}
        self.error: Optional[str] = None
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
    
    def set_context(self, chunks: List[DocumentChunk]):
        """设置上下文"""
        self.context_chunks = chunks
    
    def set_answer(self, answer: str):
        """设置回答"""
        self.answer = answer
    
    def set_sources(self, sources: List[SourceReference]):
        """设置引用来源"""
        self.sources = sources
    
    def set_error(self, error: str):
        """设置错误信息"""
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "messages": self.messages,
            "current_question": self.current_question,
            "context_chunks": self.context_chunks,
            "sources": self.sources,
            "answer": self.answer,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
            "error": self.error
        }


class RAGAgentService:
    """RAG Agent服务主类"""
    
    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        try:
            # 创建状态图
            workflow = StateGraph(AgentState)
            
            # 添加节点
            workflow.add_node("analyze_question", self._analyze_question)
            workflow.add_node("retrieve_context", self._retrieve_context)
            workflow.add_node("generate_answer", self._generate_answer)
            workflow.add_node("format_response", self._format_response)
            
            # 设置入口点
            workflow.set_entry_point("analyze_question")
            
            # 设置边
            workflow.add_edge("analyze_question", "retrieve_context")
            workflow.add_edge("retrieve_context", "generate_answer")
            workflow.add_edge("generate_answer", "format_response")
            workflow.add_edge("format_response", END)
            
            # 编译工作流
            compiled_workflow = workflow.compile()
            
            self.logger.info("LangGraph工作流构建完成")
            return compiled_workflow
            
        except Exception as e:
            self.logger.error(f"构建LangGraph工作流失败: {e}")
            raise RuntimeError(f"工作流构建失败: {e}")
    
    def _analyze_question(self, state: AgentState) -> AgentState:
        """分析用户问题"""
        try:
            start_time = time.time()
            
            # 获取当前问题
            if state.messages:
                last_message = state.messages[-1]
                if isinstance(last_message, HumanMessage):
                    state.current_question = last_message.content
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_performance("analyze_question", operation_time)
            
            self.logger.info(f"问题分析完成: {state.current_question[:50]}...")
            return state
            
        except Exception as e:
            self.logger.error(f"问题分析失败: {e}")
            state.set_error(f"问题分析失败: {e}")
            return state
    
    def _retrieve_context(self, state: AgentState) -> AgentState:
        """检索相关上下文"""
        try:
            start_time = time.time()
            
            if not state.current_question:
                state.set_error("未找到用户问题")
                return state
            
            # 使用向量服务检索相关文档片段
            context_chunks = self.vector_service.search(
                query=state.current_question,
                top_k=5
            )
            
            # 提取DocumentChunk对象
            chunks = [chunk for chunk, score in context_chunks]
            state.set_context(chunks)
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_performance("retrieve_context", operation_time, chunks_count=len(chunks))
            
            self.logger.info(f"上下文检索完成，找到{len(chunks)}个相关片段")
            return state
            
        except Exception as e:
            self.logger.error(f"上下文检索失败: {e}")
            state.set_error(f"上下文检索失败: {e}")
            return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """生成回答"""
        try:
            start_time = time.time()
            
            if not state.context_chunks:
                state.set_error("未找到相关上下文")
                return state
            
            # 使用LLM服务生成回答
            answer, sources, metadata = self.llm_service.generate_answer(
                question=state.current_question,
                context_chunks=state.context_chunks
            )
            
            state.set_answer(answer)
            state.set_sources(sources)
            state.metadata.update(metadata)
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_performance("generate_answer", operation_time, answer_length=len(answer))
            
            self.logger.info(f"回答生成完成，回答长度: {len(answer)}")
            return state
            
        except Exception as e:
            self.logger.error(f"回答生成失败: {e}")
            state.set_error(f"回答生成失败: {e}")
            return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """格式化响应"""
        try:
            start_time = time.time()
            
            # 创建AI消息
            ai_message = AIMessage(content=state.answer)
            state.add_message(ai_message)
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_performance("format_response", operation_time)
            
            self.logger.info("响应格式化完成")
            return state
            
        except Exception as e:
            self.logger.error(f"响应格式化失败: {e}")
            state.set_error(f"响应格式化失败: {e}")
            return state
    
    def process_question(
        self, 
        question: str, 
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """处理用户问题"""
        try:
            start_time = time.time()
            
            # 创建初始状态
            state = AgentState()
            state.conversation_id = conversation_id or f"conv_{int(time.time())}"
            
            # 添加用户消息
            user_message = HumanMessage(content=question)
            state.add_message(user_message)
            
            # 执行工作流
            final_state = self.workflow.invoke(state)
            
            # 构建响应
            response = {
                "conversation_id": final_state.conversation_id,
                "answer": final_state.answer,
                "sources": [source.dict() for source in final_state.sources],
                "metadata": final_state.metadata,
                "error": final_state.error,
                "success": final_state.error is None
            }
            
            # 记录总处理时间
            total_time = time.time() - start_time
            log_performance("process_question_total", total_time, question_length=len(question))
            
            self.logger.info(f"问题处理完成，总耗时: {total_time:.3f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"处理用户问题失败: {e}")
            return {
                "conversation_id": conversation_id,
                "answer": "",
                "sources": [],
                "metadata": {},
                "error": f"处理失败: {e}",
                "success": False
            }
    
    def process_conversation(
        self, 
        conversation_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """处理对话消息"""
        try:
            # 这里可以实现对话历史管理逻辑
            # 暂时直接处理单条消息
            
            return self.process_question(message, conversation_id)
            
        except Exception as e:
            self.logger.error(f"处理对话消息失败: {e}")
            return {
                "conversation_id": conversation_id,
                "answer": "",
                "sources": [],
                "metadata": {},
                "error": f"对话处理失败: {e}",
                "success": False
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        try:
            return {
                "workflow_type": "LangGraph",
                "nodes": ["analyze_question", "retrieve_context", "generate_answer", "format_response"],
                "edges": [
                    ("analyze_question", "retrieve_context"),
                    ("retrieve_context", "generate_answer"),
                    ("generate_answer", "format_response"),
                    ("format_response", "END")
                ],
                "status": "active"
            }
            
        except Exception as e:
            self.logger.error(f"获取工作流信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_workflow_config(self, config: Dict[str, Any]) -> bool:
        """更新工作流配置"""
        try:
            # 这里可以实现工作流配置更新逻辑
            # 例如动态调整检索参数、LLM参数等
            
            self.logger.info(f"工作流配置更新: {config}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新工作流配置失败: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            # 这里可以实现性能指标收集逻辑
            # 例如各节点的平均处理时间、成功率等
            
            return {
                "total_questions_processed": 0,
                "average_response_time": 0.0,
                "success_rate": 1.0,
                "vector_search_performance": self.vector_service.get_stats(),
                "llm_performance": self.llm_service.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {"status": "error", "error": str(e)}


class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)
        self.conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self, title: str = None, user_id: str = None) -> Conversation:
        """创建新对话"""
        try:
            conversation = Conversation(title=title, user_id=user_id)
            self.conversations[conversation.id] = conversation
            
            self.logger.info(f"创建新对话: {conversation.id}")
            return conversation
            
        except Exception as e:
            self.logger.error(f"创建对话失败: {e}")
            raise RuntimeError(f"创建对话失败: {e}")
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self.conversations.get(conversation_id)
    
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """添加消息到对话"""
        try:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                return False
            
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content
            )
            
            conversation.add_message(message)
            
            self.logger.info(f"添加消息到对话: {conversation_id}, 角色: {role}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加消息失败: {e}")
            return False
    
    def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """获取对话历史"""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            return conversation.messages
        return []
    
    def update_conversation_status(self, conversation_id: str, status: str) -> bool:
        """更新对话状态"""
        try:
            conversation = self.get_conversation(conversation_id)
            if not conversation:
                return False
            
            conversation.update_status(status)
            
            self.logger.info(f"更新对话状态: {conversation_id} -> {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"更新对话状态失败: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """删除对话"""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                self.logger.info(f"删除对话: {conversation_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"删除对话失败: {e}")
            return False
    
    def get_all_conversations(self, user_id: str = None) -> List[Conversation]:
        """获取所有对话"""
        if user_id:
            return [conv for conv in self.conversations.values() if conv.user_id == user_id]
        return list(self.conversations.values())
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        try:
            total_conversations = len(self.conversations)
            total_messages = sum(conv.message_count for conv in self.conversations.values())
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "average_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0,
                "active_conversations": len([conv for conv in self.conversations.values() if conv.status == "active"])
            }
            
        except Exception as e:
            self.logger.error(f"获取对话统计信息失败: {e}")
            return {"status": "error", "error": str(e)}
