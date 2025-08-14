#!/bin/bash

# RAG系统启动脚本

set -e

echo "🚀 启动RAG智能问答系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python 3.9或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装uv包管理器..."
    pip install uv
fi

echo "✅ uv包管理器已就绪"

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境变量文件..."
    cp env.example .env
    echo "⚠️  请编辑.env文件，配置必要的环境变量"
    echo "   特别是OPENAI_API_KEY和LANGCHAIN_API_KEY"
    read -p "按回车键继续..."
fi

echo "✅ 环境变量文件检查通过"

# 安装依赖
echo "📦 安装项目依赖..."
uv sync

echo "✅ 依赖安装完成"

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p data/uploads data/temp data/vector_index logs

echo "✅ 目录创建完成"

# 启动服务
echo "🚀 启动RAG服务..."
uv run python -m app.main

echo "✅ RAG系统启动完成！"
echo "📖 API文档: http://localhost:8000/docs"
echo "🔍 健康检查: http://localhost:8000/health"
echo "📊 系统统计: http://localhost:8000/api/v1/stats"
