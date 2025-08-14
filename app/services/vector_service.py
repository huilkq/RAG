"""
向量服务

支持多种向量数据库的索引构建和相似度检索
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..core.config import get_settings
from ..models.document import DocumentChunk
from ..core.logging import log_vector_operation

settings = get_settings()


class BaseVectorDB(ABC):
    """向量数据库基类"""
    
    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)
        self.embedding_model = None
        self.embedding_dimension = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化向量数据库"""
        pass
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """添加文档片段到向量数据库"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文档片段"""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        pass
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的向量嵌入"""
        if self.embedding_model is None:
            self._load_embedding_model()
        
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            self.logger.error(f"获取向量嵌入失败: {e}")
            raise RuntimeError(f"向量嵌入失败: {e}")
    
    def _load_embedding_model(self):
        """加载向量模型"""
        try:
            self.logger.info(f"加载向量模型: {settings.embedding_model}")
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"向量模型加载完成，维度: {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"加载向量模型失败: {e}")
            raise RuntimeError(f"向量模型加载失败: {e}")


class FAISSVectorDB(BaseVectorDB):
    """FAISS向量数据库"""
    
    def __init__(self):
        super().__init__()
        self.index = None
        self.chunks_map = {}  # 存储chunk_id到DocumentChunk的映射
        self.document_chunks_map = {}  # 存储document_id到chunk_ids的映射
    
    def initialize(self) -> bool:
        """初始化FAISS索引"""
        try:
            import faiss
            
            self.logger.info("初始化FAISS向量数据库")
            
            # 加载向量模型以获取维度
            self._load_embedding_model()
            
            # 创建FAISS索引
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # 内积索引
            
            # 确保索引目录存在
            Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("FAISS向量数据库初始化完成")
            return True
            
        except ImportError:
            self.logger.error("FAISS未安装，请运行: pip install faiss-cpu")
            raise RuntimeError("FAISS未安装")
        except Exception as e:
            self.logger.error(f"初始化FAISS失败: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """添加文档片段到FAISS索引"""
        try:
            if not chunks:
                return True
            
            start_time = time.time()
            self.logger.info(f"开始添加{len(chunks)}个文档片段到FAISS索引")
            
            # 获取文本内容
            texts = [chunk.content for chunk in chunks]
            
            # 获取向量嵌入
            embeddings = self._get_embeddings(texts)
            
            # 添加到FAISS索引
            self.index.add(embeddings.astype('float32'))
            
            # 更新映射关系
            for chunk in chunks:
                self.chunks_map[chunk.id] = chunk
                if chunk.document_id not in self.document_chunks_map:
                    self.document_chunks_map[chunk.document_id] = []
                self.document_chunks_map[chunk.document_id].append(chunk.id)
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "add_documents", 
                "faiss", 
                operation_time,
                chunks_count=len(chunks)
            )
            
            self.logger.info(f"成功添加{len(chunks)}个文档片段到FAISS索引，耗时: {operation_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档片段到FAISS失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文档片段"""
        try:
            start_time = time.time()
            
            # 获取查询的向量嵌入
            query_embedding = self._get_embeddings([query])
            
            # 搜索相似向量
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks_map):
                    # 获取chunk_id（这里需要维护一个索引到chunk_id的映射）
                    chunk_id = list(self.chunks_map.keys())[idx]
                    chunk = self.chunks_map[chunk_id]
                    results.append((chunk, float(score)))
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "search", 
                "faiss", 
                operation_time,
                query_length=len(query),
                results_count=len(results)
            )
            
            self.logger.info(f"FAISS搜索完成，查询: {query[:50]}..., 结果数: {len(results)}, 耗时: {operation_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS搜索失败: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档（FAISS不支持直接删除，这里标记为删除）"""
        try:
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.document_chunks_map:
                    chunk_ids = self.document_chunks_map[doc_id]
                    for chunk_id in chunk_ids:
                        if chunk_id in self.chunks_map:
                            del self.chunks_map[chunk_id]
                            deleted_count += 1
                    del self.document_chunks_map[doc_id]
            
            self.logger.info(f"标记删除{deleted_count}个文档片段")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取FAISS索引统计信息"""
        try:
            if self.index is None:
                return {"status": "not_initialized"}
            
            return {
                "status": "initialized",
                "index_type": "FAISS",
                "total_vectors": self.index.ntotal,
                "dimension": self.index.d,
                "chunks_count": len(self.chunks_map),
                "documents_count": len(self.document_chunks_map)
            }
            
        except Exception as e:
            self.logger.error(f"获取FAISS统计信息失败: {e}")
            return {"status": "error", "error": str(e)}


class MilvusVectorDB(BaseVectorDB):
    """Milvus向量数据库"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.collection = None
    
    def initialize(self) -> bool:
        """初始化Milvus连接"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            
            self.logger.info("初始化Milvus向量数据库")
            
            # 连接Milvus
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            
            # 加载向量模型以获取维度
            self._load_embedding_model()
            
            # 定义集合模式
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension)
            ]
            
            schema = CollectionSchema(fields, description="RAG文档片段向量集合")
            
            # 创建集合
            self.collection = Collection(settings.milvus_collection, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "IP",  # 内积
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            
            self.logger.info("Milvus向量数据库初始化完成")
            return True
            
        except ImportError:
            self.logger.error("PyMilvus未安装，请运行: pip install pymilvus")
            raise RuntimeError("PyMilvus未安装")
        except Exception as e:
            self.logger.error(f"初始化Milvus失败: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """添加文档片段到Milvus集合"""
        try:
            if not chunks:
                return True
            
            start_time = time.time()
            self.logger.info(f"开始添加{len(chunks)}个文档片段到Milvus集合")
            
            # 获取文本内容和向量嵌入
            texts = [chunk.content for chunk in chunks]
            embeddings = self._get_embeddings(texts)
            
            # 准备数据
            data = [
                [chunk.id for chunk in chunks],
                [chunk.document_id for chunk in chunks],
                texts,
                embeddings.tolist()
            ]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "add_documents", 
                "milvus", 
                operation_time,
                chunks_count=len(chunks)
            )
            
            self.logger.info(f"成功添加{len(chunks)}个文档片段到Milvus集合，耗时: {operation_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档片段到Milvus失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文档片段"""
        try:
            start_time = time.time()
            
            # 获取查询的向量嵌入
            query_embedding = self._get_embeddings([query])
            
            # 搜索参数
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "document_id", "content"]
            )
            
            # 构建结果
            search_results = []
            for hits in results:
                for hit in hits:
                    # 这里需要从数据库重新获取完整的DocumentChunk对象
                    # 简化处理，直接创建临时对象
                    chunk = DocumentChunk(
                        id=hit.entity.get("id"),
                        document_id=hit.entity.get("document_id"),
                        content=hit.entity.get("content"),
                        chunk_index=0,
                        start_char=0,
                        end_char=len(hit.entity.get("content", ""))
                    )
                    search_results.append((chunk, hit.score))
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "search", 
                "milvus", 
                operation_time,
                query_length=len(query),
                results_count=len(search_results)
            )
            
            self.logger.info(f"Milvus搜索完成，查询: {query[:50]}..., 结果数: {len(search_results)}, 耗时: {operation_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Milvus搜索失败: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        try:
            # 构建删除表达式
            expr = f"document_id in {document_ids}"
            self.collection.delete(expr)
            self.collection.flush()
            
            self.logger.info(f"成功删除{len(document_ids)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Milvus集合统计信息"""
        try:
            if self.collection is None:
                return {"status": "not_initialized"}
            
            stats = self.collection.get_statistics()
            return {
                "status": "initialized",
                "index_type": "Milvus",
                "collection_name": self.collection.name,
                "total_vectors": stats.get("row_count", 0),
                "dimension": self.embedding_dimension
            }
            
        except Exception as e:
            self.logger.error(f"获取Milvus统计信息失败: {e}")
            return {"status": "error", "error": str(e)}


class ChromaVectorDB(BaseVectorDB):
    """Chroma向量数据库"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.collection = None
    
    def initialize(self) -> bool:
        """初始化Chroma数据库"""
        try:
            import chromadb
            
            self.logger.info("初始化Chroma向量数据库")
            
            # 创建客户端
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
            
            # 加载向量模型以获取维度
            self._load_embedding_model()
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"description": "RAG文档片段向量集合"}
            )
            
            self.logger.info("Chroma向量数据库初始化完成")
            return True
            
        except ImportError:
            self.logger.error("ChromaDB未安装，请运行: pip install chromadb")
            raise RuntimeError("ChromaDB未安装")
        except Exception as e:
            self.logger.error(f"初始化Chroma失败: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """添加文档片段到Chroma集合"""
        try:
            if not chunks:
                return True
            
            start_time = time.time()
            self.logger.info(f"开始添加{len(chunks)}个文档片段到Chroma集合")
            
            # 获取文本内容和向量嵌入
            texts = [chunk.content for chunk in chunks]
            embeddings = self._get_embeddings(texts)
            
            # 准备数据
            ids = [chunk.id for chunk in chunks]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                }
                for chunk in chunks
            ]
            
            # 插入数据
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "add_documents", 
                "chroma", 
                operation_time,
                chunks_count=len(chunks)
            )
            
            self.logger.info(f"成功添加{len(chunks)}个文档片段到Chroma集合，耗时: {operation_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档片段到Chroma失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文档片段"""
        try:
            start_time = time.time()
            
            # 执行搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 构建结果
            search_results = []
            if results["documents"] and results["metadatas"] and results["distances"]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # 创建DocumentChunk对象
                    chunk = DocumentChunk(
                        id=results["ids"][0][i],
                        document_id=metadata.get("document_id", ""),
                        content=doc,
                        chunk_index=metadata.get("chunk_index", 0),
                        start_char=metadata.get("start_char", 0),
                        end_char=metadata.get("end_char", len(doc))
                    )
                    
                    # 将距离转换为相似度分数
                    similarity_score = 1.0 / (1.0 + distance)
                    search_results.append((chunk, similarity_score))
            
            # 记录性能日志
            operation_time = time.time() - start_time
            log_vector_operation(
                "search", 
                "chroma", 
                operation_time,
                query_length=len(query),
                results_count=len(search_results)
            )
            
            self.logger.info(f"Chroma搜索完成，查询: {query[:50]}..., 结果数: {len(search_results)}, 耗时: {operation_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Chroma搜索失败: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        try:
            # 构建删除条件
            where = {"document_id": {"$in": document_ids}}
            self.collection.delete(where=where)
            
            self.logger.info(f"成功删除{len(document_ids)}个文档")
            return True
            
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Chroma集合统计信息"""
        try:
            if self.collection is None:
                return {"status": "not_initialized"}
            
            count = self.collection.count()
            return {
                "status": "initialized",
                "index_type": "Chroma",
                "collection_name": self.collection.name,
                "total_vectors": count,
                "dimension": self.embedding_dimension
            }
            
        except Exception as e:
            self.logger.error(f"获取Chroma统计信息失败: {e}")
            return {"status": "error", "error": str(e)}


class VectorService:
    """向量服务主类"""
    
    def __init__(self):
        self.logger = logger.bind(name=self.__class__.__name__)
        self.vector_db = None
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """初始化向量数据库"""
        try:
            db_type = settings.vector_db_type.lower()
            
            if db_type == "faiss":
                self.vector_db = FAISSVectorDB()
            elif db_type == "milvus":
                self.vector_db = MilvusVectorDB()
            elif db_type == "chroma":
                self.vector_db = ChromaVectorDB()
            else:
                self.logger.warning(f"不支持的向量数据库类型: {db_type}，使用FAISS作为默认")
                self.vector_db = FAISSVectorDB()
            
            # 初始化数据库
            if not self.vector_db.initialize():
                raise RuntimeError(f"初始化向量数据库失败: {db_type}")
            
            self.logger.info(f"向量数据库初始化成功: {db_type}")
            
        except Exception as e:
            self.logger.error(f"初始化向量数据库失败: {e}")
            raise RuntimeError(f"向量数据库初始化失败: {e}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """添加文档片段到向量数据库"""
        return self.vector_db.add_documents(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """搜索相似文档片段"""
        return self.vector_db.search(query, top_k)
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        return self.vector_db.delete_documents(document_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        return self.vector_db.get_stats()
    
    def rebuild_index(self, chunks: List[DocumentChunk]) -> bool:
        """重建向量索引"""
        try:
            self.logger.info("开始重建向量索引")
            
            # 删除现有索引
            if hasattr(self.vector_db, 'index') and self.vector_db.index is not None:
                # 对于FAISS，重新创建索引
                if isinstance(self.vector_db, FAISSVectorDB):
                    import faiss
                    self.vector_db.index = faiss.IndexFlatIP(self.vector_db.embedding_dimension)
                    self.vector_db.chunks_map.clear()
                    self.vector_db.document_chunks_map.clear()
            
            # 重新添加文档
            success = self.add_documents(chunks)
            
            if success:
                self.logger.info("向量索引重建完成")
            else:
                self.logger.error("向量索引重建失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"重建向量索引失败: {e}")
            return False
