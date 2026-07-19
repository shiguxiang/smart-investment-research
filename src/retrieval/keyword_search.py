"""
关键词检索模块
基于 BM25 算法的中文关键词检索
jieba 分词 + rank-bm25 实现
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import jieba
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KeywordSearchResult:
    """关键词检索结果"""
    doc_id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)


class BM25Index:
    """
    BM25 关键词检索引擎
    支持增量构建、保存/加载、元数据过滤
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Dict] = []  # [{id, text, metadata}]
        self._tokenized_corpus: List[List[str]] = []

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._tokenized_corpus) > 0

    def add_documents(self, documents: List[Dict]):
        """
        添加文档到索引

        Args:
            documents: [{id, text, metadata}] 格式的文档列表
        """
        for doc in documents:
            self._documents.append(doc)
            tokens = self._tokenize(doc.get("text", ""))
            self._tokenized_corpus.append(tokens)

        # 重建 BM25 索引
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            f"BM25 索引已更新: 新增 {len(documents)} 篇文档, "
            f"总计 {len(self._documents)} 篇"
        )

    def build_from_chunks(self, chunks: List) -> "BM25Index":
        """
        从 TextChunk 列表构建索引

        Args:
            chunks: TextChunk 对象列表
        """
        documents = []
        for chunk in chunks:
            documents.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
        self.add_documents(documents)
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict] = None,
    ) -> List[KeywordSearchResult]:
        """
        执行关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            metadata_filter: 元数据过滤条件 (如 {"subject": "高等数学"})

        Returns:
            KeywordSearchResult 列表，按分数降序
        """
        if not self.is_ready:
            logger.warning("BM25 索引未就绪")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 构建结果列表
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue

            doc = self._documents[idx]

            # 元数据过滤
            if metadata_filter:
                if not self._match_filter(doc.get("metadata", {}), metadata_filter):
                    continue

            results.append(KeywordSearchResult(
                doc_id=doc["id"],
                text=doc["text"],
                score=float(score),
                metadata=doc.get("metadata", {}),
            ))

        # 按分数降序
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def search_all_fields(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict] = None,
    ) -> List[KeywordSearchResult]:
        """
        增强检索: 同时搜索 text 和 metadata 字段
        """
        # 基础搜索
        results = self.search(query, top_k, metadata_filter)

        # 如果没有元数据过滤，额外按文件名/科目搜
        if not metadata_filter:
            query_lower = query.lower()
            for doc in self._documents:
                if len(results) >= top_k:
                    break
                meta = doc.get("metadata", {})
                file_name = meta.get("file_name", "").lower()
                subject = meta.get("subject", "").lower()

                if query_lower in file_name or query_lower in subject:
                    # 检查是否已在结果中
                    if not any(r.doc_id == doc["id"] for r in results):
                        results.append(KeywordSearchResult(
                            doc_id=doc["id"],
                            text=doc["text"],
                            score=0.3,  # 元数据匹配给较低分
                            metadata=meta,
                        ))

        return results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词"""
        # 精确模式分词
        tokens = jieba.lcut(text)
        # 过滤空白和纯标点
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]
        return tokens

    @staticmethod
    def _match_filter(metadata: Dict, filter_dict: Dict) -> bool:
        """检查元数据是否满足过滤条件"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save(self, file_path: str):
        """保存索引到磁盘"""
        data = {
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 索引已保存: {file_path} ({len(self._documents)} 篇文档)")

    def load(self, file_path: str) -> bool:
        """从磁盘加载索引"""
        if not os.path.exists(file_path):
            logger.warning(f"索引文件不存在: {file_path}")
            return False

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self._documents = data["documents"]
        self._tokenized_corpus = data["tokenized_corpus"]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 索引已加载: {file_path} ({len(self._documents)} 篇文档)")
        return True


# 全局索引实例
bm25_index = BM25Index()
