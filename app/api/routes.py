"""
API路由定义

包含文档管理、向量检索、智能问答等接口
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger

from ..core.config import get_settings
from ..core.logging import log_api_request
from ..models.document import (
    Document, DocumentUploadResponse, DocumentQueryRequest, 
    DocumentListResponse, DocumentProcessingRequest, DocumentProcessingResponse
)
from ..models.chat import (
    ChatQueryRequest, ChatResponse, ConversationRequest, 
    ConversationResponse, ConversationListRequest, ConversationListResponse
)
from ..services.parser_service import DocumentParserService
from ..services.vector_service import VectorService
from ..services.llm_service import LLMService
from ..services.agent_service import RAGAgentService, ConversationManager

settings = get_settings()

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["RAG API"])

# 初始化服务
parser_service = DocumentParserService()
vector_service = VectorService()
llm_service = LLMService()
agent_service = RAGAgentService()
conversation_manager = ConversationManager()


# 文档管理接口
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None)
):
    """上传文档"""
    start_time = datetime.now()
    
    try:
        # 验证文件
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 检查文件格式
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.docx', '.md', '.txt']:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
        
        # 检查文件大小
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()
        file.file.seek(0)  # 重置到文件开头
        
        if file_size > settings.max_file_size:
            raise HTTPException(status_code=400, detail="文件大小超过限制")
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        file_path = Path(settings.upload_dir) / safe_filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建文档对象
        document = Document(
            filename=safe_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file_size,
            file_type=file_ext[1:],  # 移除点号
            content="",  # 内容将在后台任务中解析
            status="uploaded"
        )
        
        # 设置元数据
        if title:
            document.metadata.title = title
        if category:
            document.metadata.category = category
        if tags:
            document.metadata.tags = [tag.strip() for tag in tags.split(',')]
        
        # 添加后台任务：解析文档并构建向量索引
        background_tasks.add_task(
            process_document_background,
            document.id,
            str(file_path),
            document.file_type
        )
        
        # 记录API请求日志
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/documents/upload", 200, duration)
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            status="uploaded",
            message="文档上传成功，正在后台处理",
            created_at=document.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/documents/upload", 500, duration)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {e}")


async def process_document_background(document_id: str, file_path: str, file_type: str):
    """后台处理文档"""
    try:
        logger.info(f"开始后台处理文档: {document_id}")
        
        # 解析文档
        content, chunks = parser_service.parse_and_chunk(document_id, file_path)
        
        # 构建向量索引
        success = vector_service.add_documents(chunks)
        
        if success:
            logger.info(f"文档处理完成: {document_id}, 片段数: {len(chunks)}")
        else:
            logger.error(f"文档处理失败: {document_id}")
            
    except Exception as e:
        logger.error(f"后台处理文档失败: {document_id}, 错误: {e}")


@router.post("/documents/build-index", response_model=DocumentProcessingResponse)
async def build_vector_index(request: DocumentProcessingRequest):
    """构建向量索引"""
    start_time = datetime.now()
    
    try:
        # 这里应该实现从存储中获取文档片段的逻辑
        # 暂时返回成功响应
        task_id = f"task_{int(datetime.now().timestamp())}"
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/documents/build-index", 200, duration)
        
        return DocumentProcessingResponse(
            task_id=task_id,
            total_documents=len(request.document_ids),
            status="processing",
            message="向量索引构建任务已启动",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"构建向量索引失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/documents/build-index", 500, duration)
        raise HTTPException(status_code=500, detail=f"构建向量索引失败: {e}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    category: str = None,
    status: str = None
):
    """获取文档列表"""
    start_time = datetime.now()
    
    try:
        # 这里应该实现从存储中获取文档列表的逻辑
        # 暂时返回空列表
        documents = []
        total = 0
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", "/api/v1/documents", 200, duration)
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=0
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", "/api/v1/documents", 500, duration)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {e}")


# 智能问答接口
@router.post("/chat/query", response_model=ChatResponse)
async def chat_query(request: ChatQueryRequest):
    """智能问答"""
    start_time = datetime.now()
    
    try:
        # 使用Agent服务处理问题
        response = agent_service.process_question(
            question=request.question,
            conversation_id=request.conversation_id
        )
        
        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])
        
        # 构建响应
        chat_response = ChatResponse(
            answer=response["answer"],
            conversation_id=response["conversation_id"],
            message_id=f"msg_{int(datetime.now().timestamp())}",
            sources=response["sources"],
            confidence=0.9,  # 这里可以基于相似度分数计算
            tokens_used=response["metadata"].get("tokens_used", {}),
            processing_time=response["metadata"].get("processing_time", 0),
            timestamp=datetime.now()
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/chat/query", 200, duration)
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"智能问答失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/chat/query", 500, duration)
        raise HTTPException(status_code=500, detail=f"智能问答失败: {e}")


@router.post("/chat/conversation", response_model=ConversationResponse)
async def conversation_chat(request: ConversationRequest):
    """对话聊天"""
    start_time = datetime.now()
    
    try:
        # 使用Agent服务处理对话
        response = agent_service.process_conversation(
            conversation_id=request.conversation_id,
            message=request.message
        )
        
        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])
        
        # 获取对话历史
        conversation = conversation_manager.get_conversation(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/chat/conversation", 200, duration)
        
        return ConversationResponse(
            conversation_id=conversation.id,
            messages=conversation.messages,
            total_messages=conversation.message_count,
            last_updated=conversation.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话聊天失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/chat/conversation", 500, duration)
        raise HTTPException(status_code=500, detail=f"对话聊天失败: {e}")


# 对话管理接口
@router.post("/conversations")
async def create_conversation(title: str = None, user_id: str = None):
    """创建新对话"""
    start_time = datetime.now()
    
    try:
        conversation = conversation_manager.create_conversation(title=title, user_id=user_id)
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/conversations", 200, duration)
        
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at
        }
        
    except Exception as e:
        logger.error(f"创建对话失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/conversations", 500, duration)
        raise HTTPException(status_code=500, detail=f"创建对话失败: {e}")


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    user_id: str = None,
    status: str = None
):
    """获取对话列表"""
    start_time = datetime.now()
    
    try:
        conversations = conversation_manager.get_all_conversations(user_id=user_id)
        
        # 分页处理
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_conversations = conversations[start_idx:end_idx]
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", "/api/v1/conversations", 200, duration)
        
        return ConversationListResponse(
            conversations=paginated_conversations,
            total=len(conversations),
            page=page,
            page_size=page_size,
            total_pages=(len(conversations) + page_size - 1) // page_size
        )
        
    except Exception as e:
        logger.error(f"获取对话列表失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", "/api/v1/conversations", 500, duration)
        raise HTTPException(status_code=500, detail=f"获取对话列表失败: {e}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取对话详情"""
    start_time = datetime.now()
    
    try:
        conversation = conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", f"/api/v1/conversations/{conversation_id}", 200, duration)
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话详情失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("GET", f"/api/v1/conversations/{conversation_id}", 500, duration)
        raise HTTPException(status_code=500, detail=f"获取对话详情失败: {e}")


# 系统状态接口
@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="服务异常")


@router.get("/stats")
async def get_system_stats():
    """获取系统统计信息"""
    try:
        vector_stats = vector_service.get_stats()
        llm_info = llm_service.get_model_info()
        agent_info = agent_service.get_workflow_info()
        conversation_stats = conversation_manager.get_conversation_stats()
        
        return {
            "vector_database": vector_stats,
            "llm_service": llm_info,
            "agent_workflow": agent_info,
            "conversations": conversation_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {e}")


# 向量搜索接口
@router.post("/search")
async def vector_search(query: str, top_k: int = 5):
    """向量搜索"""
    start_time = datetime.now()
    
    try:
        # 执行向量搜索
        results = vector_service.search(query, top_k)
        
        # 格式化结果
        search_results = []
        for chunk, score in results:
            search_results.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "similarity_score": score,
                "metadata": chunk.metadata
            })
        
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/search", 200, duration)
        
        return {
            "query": query,
            "results": search_results,
            "total_results": len(search_results),
            "search_time": duration
        }
        
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        log_api_request("POST", "/api/v1/search", 500, duration)
        raise HTTPException(status_code=500, detail=f"向量搜索失败: {e}")


# 工具接口
@router.post("/tools/extract-keywords")
async def extract_keywords(text: str, max_keywords: int = 10):
    """提取关键词"""
    try:
        keywords = llm_service.extract_keywords(text, max_keywords)
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "keywords": keywords,
            "count": len(keywords)
        }
        
    except Exception as e:
        logger.error(f"提取关键词失败: {e}")
        raise HTTPException(status_code=500, detail=f"提取关键词失败: {e}")


@router.post("/tools/classify-document")
async def classify_document(content: str, categories: List[str]):
    """文档分类"""
    try:
        category = llm_service.classify_document(content, categories)
        
        return {
            "content": content[:200] + "..." if len(content) > 200 else content,
            "categories": categories,
            "classified_category": category
        }
        
    except Exception as e:
        logger.error(f"文档分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档分类失败: {e}")


@router.post("/tools/summarize-conversation")
async def summarize_conversation(conversation_history: List[Dict[str, str]]):
    """总结对话"""
    try:
        summary = llm_service.summarize_conversation(conversation_history)
        
        return {
            "conversation_history": conversation_history,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"总结对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"总结对话失败: {e}")
