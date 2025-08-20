# RAG 智能问答系统

基于 LangGraph 构建的 RAG (Retrieval-Augmented Generation) 系统，支持文档上传、向量检索和智能问答。

## 功能特性

- 📄 **多格式文档支持**: PDF、Word (.docx)、Markdown (.md)、TXT
- 🔍 **智能文本分段**: 自动进行合适粒度的文本分段
- 🧠 **向量化索引**: 支持 FAISS、Milvus、Chroma 等向量数据库
- 💬 **智能问答**: 基于向量检索的 LLM 问答，支持原文引用
- 🗣️ **多轮对话**: 保持上下文，支持连续提问和补充提问
- 📊 **可观测性**: 集成 LangSmith，提供完整的调用链追踪
- 🚀 **LangGraph 集成**: 使用 LangGraph 构建 Agent 执行流程

## 技术栈

- **Web 框架**: FastAPI
- **LLM 框架**: LangChain + LangGraph
- **向量数据库**: FAISS (默认) / Milvus / Chroma
- **文档解析**: PyPDF, python-docx, markdown
- **向量模型**: Sentence Transformers
- **日志系统**: Loguru
- **依赖管理**: uv
- **容器化**: Docker + Docker Compose

## 快速开始

### 环境要求

- Python 3.9+
- Docker & Docker Compose (可选)

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置必要的环境变量：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# LangSmith 配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=rag-system

# 向量数据库配置
VECTOR_DB_TYPE=faiss  # faiss, milvus, chroma
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis 配置
REDIS_URL=redis://localhost:6379
```

### 启动服务

#### 方式一：直接启动

```bash
# 启动主服务
uv run python -m app.main

# 启动向量索引服务
uv run python -m app.services.vector_service
```

#### 方式二：Docker 启动

```bash
# 构建镜像
docker build -t rag-system .

# 启动所有服务
docker-compose up -d
```

### 使用示例

#### 1. 上传文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.pdf" \
  -F "metadata={\"title\":\"示例文档\",\"category\":\"技术文档\"}"
```

#### 2. 构建向量索引

```bash
curl -X POST "http://localhost:8000/api/v1/documents/build-index" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc_123"]}'
```

#### 3. 智能问答

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是RAG系统？",
    "conversation_id": "conv_123",
    "use_context": true
  }'
```

#### 4. 多轮对话

```bash
curl -X POST "http://localhost:8000/api/v1/chat/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "message": "能详细解释一下吗？"
  }'
```

## 项目结构

```
RAG/
├── app/                    # 主应用代码
│   ├── api/               # API 路由
│   ├── core/              # 核心配置
│   ├── models/            # 数据模型
│   ├── services/          # 业务服务
│   ├── utils/             # 工具函数
│   └── main.py            # 应用入口
├── tests/                 # 测试代码
├── docker/                # Docker 配置
├── scripts/               # 启动脚本
├── docs/                  # 项目文档
├── pyproject.toml         # 项目配置
├── Dockerfile             # Docker 镜像
├── docker-compose.yml     # 服务编排
└── README.md              # 项目说明
```

## 开发指南


### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_document_service.py

# 生成覆盖率报告
uv run pytest --cov=app tests/
```

## 部署说明

### Docker 部署

1. 构建镜像：
   ```bash
   docker build -t rag-system .
   ```

2. 启动服务：
   ```bash
   docker-compose up -d
   ```

3. 检查服务状态：
   ```bash
   docker-compose ps
   ```
