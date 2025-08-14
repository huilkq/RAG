"""
RAG系统主应用入口

基于FastAPI构建的RAG智能问答系统
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .core.config import get_settings
from .core.logging import setup_logging
from .api.routes import router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("RAG系统启动中...")
    
    # 确保必要的目录存在
    import os
    from pathlib import Path
    
    directories = [
        settings.upload_dir,
        settings.temp_dir,
        settings.vector_db_path,
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {directory}")
    
    logger.info("RAG系统启动完成")
    
    yield
    
    # 关闭时执行
    logger.info("RAG系统关闭中...")
    logger.info("RAG系统已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于LangGraph的RAG智能问答系统",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 添加请求处理中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# 添加异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc) if settings.debug else "请联系管理员",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


# 注册路由
app.include_router(router)


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG智能问答系统",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "stats": "/api/v1/stats"
    }


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "RAG System",
        "version": settings.app_version,
        "timestamp": time.time()
    }


# 启动事件
@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("RAG系统正在启动...")
    
    # 初始化日志系统
    setup_logging()
    
    # 记录启动信息
    logger.info(f"应用名称: {settings.app_name}")
    logger.info(f"应用版本: {settings.app_version}")
    logger.info(f"调试模式: {settings.debug}")
    logger.info(f"日志级别: {settings.log_level}")
    logger.info(f"向量数据库类型: {settings.vector_db_type}")
    logger.info(f"LLM模型: {settings.openai_model}")
    
    if settings.langchain_tracing_v2:
        logger.info("LangSmith追踪已启用")
    else:
        logger.info("LangSmith追踪未启用")


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("RAG系统正在关闭...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
