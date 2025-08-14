# RAG系统快速启动指南

## 🚀 5分钟快速启动

本指南将帮助您在5分钟内启动RAG智能问答系统。

## 前置要求

- Python 3.9+
- OpenAI API密钥
- 可选：LangSmith API密钥

## 快速启动步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd RAG
```

### 2. 安装依赖
```bash
# 使用uv（推荐）
pip install uv
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制环境变量模板
cp env.example .env

# 编辑.env文件，配置OpenAI API密钥
nano .env
```

**必需配置**：
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 启动服务
```bash
# 使用启动脚本
chmod +x scripts/start.sh
./scripts/start.sh

# 或直接启动
uv run python -m app.main
```

### 5. 验证启动
```bash
# 健康检查
curl http://localhost:8000/health

# 查看API文档
# 浏览器访问: http://localhost:8000/docs
```

## 🐳 Docker快速启动

### 1. 使用Docker Compose
```bash
# 配置环境变量
cp env.example .env
# 编辑.env文件

# 启动服务
docker-compose up -d

# 查看状态
docker-compose ps
```

### 2. 使用Docker启动脚本
```bash
chmod +x scripts/docker-start.sh
./scripts/docker-start.sh
```

## 📖 快速体验

### 1. 上传文档
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf" \
  -F "title=示例文档"
```

### 2. 智能问答
```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是RAG系统？",
    "use_context": true
  }'
```

### 3. 向量搜索
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG技术",
    "top_k": 5
  }'
```

## 🔧 常见问题

### Q: 启动失败怎么办？
A: 检查以下几点：
- Python版本是否为3.9+
- 是否配置了OPENAI_API_KEY
- 端口8000是否被占用

### Q: 如何更换向量数据库？
A: 修改.env文件中的VECTOR_DB_TYPE：
```bash
VECTOR_DB_TYPE=faiss    # 或 milvus, chroma
```

### Q: 如何查看日志？
A: 查看logs目录下的日志文件：
```bash
tail -f logs/app.log
```

## 📚 下一步

- 阅读[完整README](README.md)了解详细功能
- 查看[部署说明](docs/部署说明.md)了解生产部署
- 探索[项目概述](docs/项目概述.md)了解系统架构

## 🆘 获取帮助

- 查看[API文档](http://localhost:8000/docs)
- 检查[系统状态](http://localhost:8000/api/v1/stats)
- 查看日志文件了解详细错误信息

---

🎉 **恭喜！您已成功启动RAG系统！**

现在可以开始使用智能问答功能了。如有问题，请参考完整文档或提交Issue。
