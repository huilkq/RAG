#!/bin/bash

# RAG系统Docker启动脚本

set -e

echo "🐳 启动RAG系统Docker服务..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

echo "✅ Docker环境检查通过"

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境变量文件..."
    cp env.example .env
    echo "⚠️  请编辑.env文件，配置必要的环境变量"
    echo "   特别是OPENAI_API_KEY和LANGCHAIN_API_KEY"
    read -p "按回车键继续..."
fi

echo "✅ 环境变量文件检查通过"

# 构建镜像
echo "🔨 构建Docker镜像..."
docker build -t rag-system .

echo "✅ 镜像构建完成"

# 启动服务
echo "🚀 启动Docker服务..."
docker-compose up -d

echo "✅ Docker服务启动完成！"

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "📊 检查服务状态..."
docker-compose ps

# 显示访问信息
echo ""
echo "🎉 RAG系统启动成功！"
echo "📖 API文档: http://localhost:8000/docs"
echo "🔍 健康检查: http://localhost:8000/health"
echo "📊 系统统计: http://localhost:8000/api/v1/stats"
echo ""
echo "🐳 Docker服务管理命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  查看状态: docker-compose ps"
