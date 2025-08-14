"""
文档数据模型

定义文档、文档片段等核心数据结构
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentMetadata(BaseModel):
    """文档元数据"""
    
    title: Optional[str] = Field(default=None, description="文档标题")
    author: Optional[str] = Field(default=None, description="文档作者")
    category: Optional[str] = Field(default=None, description="文档分类")
    tags: List[str] = Field(default_factory=list, description="文档标签")
    source: Optional[str] = Field(default=None, description="文档来源")
    language: str = Field(default="zh-CN", description="文档语言")
    created_at: Optional[datetime] = Field(default=None, description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="自定义字段")


class DocumentChunk(BaseModel):
    """文档片段"""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="片段ID")
    document_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="片段内容")
    chunk_index: int = Field(..., description="片段在文档中的索引")
    start_char: int = Field(..., description="起始字符位置")
    end_char: int = Field(..., description="结束字符位置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="片段元数据")
    embedding: Optional[List[float]] = Field(default=None, description="向量嵌入")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    
    @validator("content")
    def validate_content(cls, v):
        """验证内容不为空"""
        if not v or not v.strip():
            raise ValueError("片段内容不能为空")
        return v.strip()
    
    @property
    def length(self) -> int:
        """获取片段长度"""
        return len(self.content)


class Document(BaseModel):
    """文档模型"""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="文档ID")
    filename: str = Field(..., description="文件名")
    original_filename: str = Field(..., description="原始文件名")
    file_path: str = Field(..., description="文件存储路径")
    file_size: int = Field(..., description="文件大小(字节)")
    file_type: str = Field(..., description="文件类型")
    content: str = Field(..., description="文档内容")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="文档元数据")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="文档片段")
    status: str = Field(default="uploaded", description="文档状态")
    processing_status: str = Field(default="pending", description="处理状态")
    vector_indexed: bool = Field(default=False, description="是否已建立向量索引")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    
    @validator("file_type")
    def validate_file_type(cls, v):
        """验证文件类型"""
        supported_types = ["pdf", "docx", "md", "txt"]
        if v.lower() not in supported_types:
            raise ValueError(f"不支持的文件类型: {v}")
        return v.lower()
    
    @validator("status")
    def validate_status(cls, v):
        """验证文档状态"""
        valid_statuses = ["uploaded", "processing", "processed", "error", "deleted"]
        if v not in valid_statuses:
            raise ValueError(f"无效的文档状态: {v}")
        return v
    
    @validator("processing_status")
    def validate_processing_status(cls, v):
        """验证处理状态"""
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if v not in valid_statuses:
            raise ValueError(f"无效的处理状态: {v}")
        return v
    
    @property
    def chunks_count(self) -> int:
        """获取片段数量"""
        return len(self.chunks)
    
    @property
    def total_content_length(self) -> int:
        """获取总内容长度"""
        return len(self.content)
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """添加文档片段"""
        chunk.document_id = self.id
        self.chunks.append(chunk)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """根据ID获取片段"""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def update_status(self, new_status: str) -> None:
        """更新文档状态"""
        self.status = new_status
        self.updated_at = datetime.utcnow()
    
    def update_processing_status(self, new_status: str) -> None:
        """更新处理状态"""
        self.processing_status = new_status
        self.updated_at = datetime.utcnow()


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小")
    metadata: Optional[DocumentMetadata] = Field(default=None, description="文档元数据")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    
    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    status: str = Field(..., description="上传状态")
    message: str = Field(..., description="响应消息")
    created_at: datetime = Field(..., description="创建时间")


class DocumentQueryRequest(BaseModel):
    """文档查询请求"""
    
    document_id: Optional[str] = Field(default=None, description="文档ID")
    filename: Optional[str] = Field(default=None, description="文件名")
    category: Optional[str] = Field(default=None, description="文档分类")
    tags: Optional[List[str]] = Field(default=None, description="文档标签")
    status: Optional[str] = Field(default=None, description="文档状态")
    page: int = Field(default=1, description="页码")
    page_size: int = Field(default=20, description="每页大小")
    
    @validator("page")
    def validate_page(cls, v):
        """验证页码"""
        if v < 1:
            raise ValueError("页码必须大于0")
        return v
    
    @validator("page_size")
    def validate_page_size(cls, v):
        """验证每页大小"""
        if v < 1 or v > 100:
            raise ValueError("每页大小必须在1-100之间")
        return v


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    
    documents: List[Document] = Field(..., description="文档列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total_pages: int = Field(..., description="总页数")


class DocumentProcessingRequest(BaseModel):
    """文档处理请求"""
    
    document_ids: List[str] = Field(..., description="要处理的文档ID列表")
    force_reprocess: bool = Field(default=False, description="是否强制重新处理")


class DocumentProcessingResponse(BaseModel):
    """文档处理响应"""
    
    task_id: str = Field(..., description="任务ID")
    total_documents: int = Field(..., description="总文档数")
    status: str = Field(..., description="处理状态")
    message: str = Field(..., description="响应消息")
    created_at: datetime = Field(..., description="创建时间")
