"""
日志配置模块

使用 loguru 进行日志管理，确保不使用 print 语句
"""

import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from loguru._defaults import LOGURU_DEFAULT_CONFIG

from .config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """配置日志系统"""
    
    # 移除默认的日志处理器
    logger.remove()
    
    # 配置控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 配置文件输出
    log_file = Path("./logs/app.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 配置错误日志文件
    error_log_file = Path("./logs/error.log")
    logger.add(
        str(error_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # 配置性能日志文件
    perf_log_file = Path("./logs/performance.log")
    logger.add(
        str(perf_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        filter=lambda record: "performance" in record["extra"],
        rotation="50 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("日志系统初始化完成")


def get_logger(name: str = None) -> "logger":
    """获取日志记录器"""
    if name:
        return logger.bind(name=name)
    return logger


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """记录性能日志"""
    logger.bind(performance=True).info(
        f"性能监控 - {operation} 耗时: {duration:.3f}s",
        extra={"operation": operation, "duration": duration, **kwargs}
    )


def log_api_request(
    method: str, 
    path: str, 
    status_code: int, 
    duration: float, 
    **kwargs
) -> None:
    """记录API请求日志"""
    logger.info(
        f"API请求 - {method} {path} | 状态: {status_code} | 耗时: {duration:.3f}s",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            **kwargs
        }
    )


def log_document_processing(
    file_name: str, 
    file_size: int, 
    processing_time: float, 
    chunks_count: int,
    **kwargs
) -> None:
    """记录文档处理日志"""
    logger.info(
        f"文档处理 - {file_name} | 大小: {file_size} bytes | "
        f"处理时间: {processing_time:.3f}s | 分段数: {chunks_count}",
        extra={
            "file_name": file_name,
            "file_size": file_size,
            "processing_time": processing_time,
            "chunks_count": chunks_count,
            **kwargs
        }
    )


def log_vector_operation(
    operation: str, 
    collection: str, 
    operation_time: float, 
    **kwargs
) -> None:
    """记录向量操作日志"""
    logger.info(
        f"向量操作 - {operation} | 集合: {collection} | 耗时: {operation_time:.3f}s",
        extra={
            "operation": operation,
            "collection": collection,
            "operation_time": operation_time,
            **kwargs
        }
    )


def log_llm_call(
    model: str, 
    prompt_tokens: int, 
    completion_tokens: int, 
    total_time: float,
    **kwargs
) -> None:
    """记录LLM调用日志"""
    logger.info(
        f"LLM调用 - {model} | 输入tokens: {prompt_tokens} | "
        f"输出tokens: {completion_tokens} | 总耗时: {total_time:.3f}s",
        extra={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time": total_time,
            **kwargs
        }
    )


# 初始化日志系统
setup_logging()
