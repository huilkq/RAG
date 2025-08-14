"""
对话数据模型

定义对话、消息等核心数据结构
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """对话消息"""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="消息ID")
    conversation_id: str = Field(..., description="对话ID")
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="消息元数据")
    
    @validator("role")
    def validate_role(cls, v):
        """验证消息角色"""
        valid_roles = ["user", "assistant", "system"]
        if v not in valid_roles:
            raise ValueError(f"无效的消息角色: {v}")
        return v
    
    @validator("content")
    def validate_content(cls, v):
        """验证消息内容"""
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()


class Conversation(BaseModel):
    """对话会话"""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="对话ID")
    title: Optional[str] = Field(default=None, description="对话标题")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    messages: List[Message] = Field(default_factory=list, description="消息列表")
    status: str = Field(default="active", description="对话状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="对话元数据")
    
    @validator("status")
    def validate_status(cls, v):
        """验证对话状态"""
        valid_statuses = ["active", "archived", "deleted"]
        if v not in valid_statuses:
            raise ValueError(f"无效的对话状态: {v}")
        return v
    
    @property
    def message_count(self) -> int:
        """获取消息数量"""
        return len(self.messages)
    
    @property
    def last_message(self) -> Optional[Message]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def add_message(self, message: Message) -> None:
        """添加消息"""
        message.conversation_id = self.id
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """根据角色获取消息"""
        return [msg for msg in self.messages if msg.role == role]
    
    def update_status(self, new_status: str) -> None:
        """更新对话状态"""
        self.status = new_status
        self.updated_at = datetime.utcnow()


class ChatQueryRequest(BaseModel):
    """聊天查询请求"""
    
    question: str = Field(..., description="用户问题")
    conversation_id: Optional[str] = Field(default=None, description="对话ID")
    use_context: bool = Field(default=True, description="是否使用上下文")
    max_context_messages: int = Field(default=5, description="最大上下文消息数")
    temperature: Optional[float] = Field(default=None, description="生成温度")
    max_tokens: Optional[int] = Field(default=None, description="最大生成token数")
    
    @validator("question")
    def validate_question(cls, v):
        """验证问题不为空"""
        if not v or not v.strip():
            raise ValueError("问题不能为空")
        return v.strip()
    
    @validator("max_context_messages")
    def validate_max_context_messages(cls, v):
        """验证最大上下文消息数"""
        if v < 1 or v > 20:
            raise ValueError("最大上下文消息数必须在1-20之间")
        return v


class ChatResponse(BaseModel):
    """聊天响应"""
    
    answer: str = Field(..., description="AI回答")
    conversation_id: str = Field(..., description="对话ID")
    message_id: str = Field(..., description="消息ID")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源")
    confidence: float = Field(..., description="置信度")
    tokens_used: Dict[str, int] = Field(default_factory=dict, description="使用的token数")
    processing_time: float = Field(..., description="处理时间(秒)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


class ConversationRequest(BaseModel):
    """对话请求"""
    
    conversation_id: str = Field(..., description="对话ID")
    message: str = Field(..., description="用户消息")
    use_context: bool = Field(default=True, description="是否使用上下文")
    
    @validator("message")
    def validate_message(cls, v):
        """验证消息不为空"""
        if not v or not v.strip():
            raise ValueError("消息不能为空")
        return v.strip()


class ConversationResponse(BaseModel):
    """对话响应"""
    
    conversation_id: str = Field(..., description="对话ID")
    messages: List[Message] = Field(..., description="消息列表")
    total_messages: int = Field(..., description="总消息数")
    last_updated: datetime = Field(..., description="最后更新时间")


class ConversationListRequest(BaseModel):
    """对话列表请求"""
    
    user_id: Optional[str] = Field(default=None, description="用户ID")
    status: Optional[str] = Field(default=None, description="对话状态")
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


class ConversationListResponse(BaseModel):
    """对话列表响应"""
    
    conversations: List[Conversation] = Field(..., description="对话列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total_pages: int = Field(..., description="总页数")


class SourceReference(BaseModel):
    """引用来源"""
    
    document_id: str = Field(..., description="文档ID")
    document_title: str = Field(..., description="文档标题")
    chunk_id: str = Field(..., description="片段ID")
    content: str = Field(..., description="引用内容")
    similarity_score: float = Field(..., description="相似度分数")
    start_char: int = Field(..., description="起始字符位置")
    end_char: int = Field(..., description="结束字符位置")


class ChatHistory(BaseModel):
    """聊天历史"""
    
    conversation_id: str = Field(..., description="对话ID")
    messages: List[Message] = Field(..., description="消息列表")
    created_at: datetime = Field(..., description="创建时间")
    last_activity: datetime = Field(..., description="最后活动时间")
    
    @property
    def duration(self) -> float:
        """获取对话持续时间(秒)"""
        return (self.last_activity - self.created_at).total_seconds()
    
    @property
    def user_message_count(self) -> int:
        """获取用户消息数量"""
        return len([msg for msg in self.messages if msg.role == "user"])
    
    @property
    def assistant_message_count(self) -> int:
        """获取助手消息数量"""
        return len([msg for msg in self.messages if msg.role == "assistant"])
