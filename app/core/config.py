"""
应用配置管理模块

使用 Pydantic 管理环境变量和配置项
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# 加载环境变量
load_dotenv()


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = Field(default="RAG System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # OpenAI 配置
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # LangSmith 配置
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="rag-system", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", 
        env="LANGCHAIN_ENDPOINT"
    )
    
    # 向量数据库配置
    vector_db_type: str = Field(default="faiss", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field(default="./data/vector_index", env="VECTOR_DB_PATH")
    
    # Milvus 配置
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_collection: str = Field(default="rag_documents", env="MILVUS_COLLECTION")
    
    # Chroma 配置
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", 
        env="CHROMA_PERSIST_DIRECTORY"
    )
    
    # Redis 配置
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # 文档处理配置
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "md", "txt"], 
        env="SUPPORTED_FORMATS"
    )
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # 向量模型配置
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # 对话配置
    max_conversation_length: int = Field(default=10, env="MAX_CONVERSATION_LENGTH")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # 存储配置
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="./data/temp", env="TEMP_DIR")
    
    @validator("supported_formats", pre=True)
    def parse_supported_formats(cls, v):
        """解析支持的文件格式"""
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(",")]
        return v
    
    @validator("upload_dir", "temp_dir", "vector_db_path", "chroma_persist_directory")
    def create_directories(cls, v):
        """确保目录存在"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
