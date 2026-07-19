"""
向量检索模块
基于 Milvus 的语义向量检索
支持标量过滤 (按科目/章节/文件类型)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from pymilvus import Collection, MilvusException

from src.retrieval.milvus_client import milvus_client, EMBEDDING_DIM
from src.retrieval.embedder import Embedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """向量检索结果"""
    doc_id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)


class VectorSearchEngine:
    """
    向量检索引擎
    封装 Milvus 检索 + DashScope Embedding
    """

    # Milvus 搜索参数
    SEARCH_PARAMS = {
        "metric_type": "IP",  # Inner Product
        "params": {"nprobe": 16},
    }

    # 输出字段
    OUTPUT_FIELDS = [
        "id", "text", "file_name", "file_type",
        "subject", "chapter", "chunk_index", "char_count", "source",
    ]

    def __init__(self):
        self._embedder = Embedder()

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection: Optional[Collection] = None,
    ) -> List[VectorSearchResult]:
        """
        执行向量检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            metadata_filter: 元数据过滤 (如 {"subject": "线性代数", "file_type": "pdf"})
            collection: 指定 Collection (默认从 milvus_client 获取)

        Returns:
            VectorSearchResult 列表，按相似度降序
        """
        # 1. 向量化查询
        query_embedding = self._embedder.embed(query)
        if query_embedding is None:
            logger.error("查询向量化失败")
            return []

        # 2. 获取 Collection
        col = collection or milvus_client.get_collection()
        if col is None:
            logger.error("Milvus Collection 不可用")
            return []

        # 3. 构建过滤表达式
        expr = self._build_filter_expr(metadata_filter)

        # 4. 执行搜索
        try:
            milvus_client.ensure_loaded(col)

            results = col.search(
                data=[query_embedding],
                anns_field="embedding",
                param=self.SEARCH_PARAMS,
                limit=top_k,
                expr=expr,
                output_fields=self.OUTPUT_FIELDS,
            )

        except MilvusException as e:
            logger.error(f"Milvus 向量检索异常: {e}")
            return []

        # 5. 解析结果
        search_results = []
        if results and results[0]:
            for hit in results[0]:
                result = VectorSearchResult(
                    doc_id=str(hit.id),
                    text=str(hit.entity.get("text", "")),
                    score=float(hit.score),
                    metadata={
                        "file_name": str(hit.entity.get("file_name", "")),
                        "file_type": str(hit.entity.get("file_type", "")),
                        "subject": str(hit.entity.get("subject", "")),
                        "chapter": str(hit.entity.get("chapter", "")),
                        "chunk_index": hit.entity.get("chunk_index", 0),
                        "char_count": hit.entity.get("char_count", 0),
                        "source": str(hit.entity.get("source", "")),
                    },
                )
                search_results.append(result)

        logger.debug(
            f"向量检索: query='{query[:50]}...', "
            f"filter={expr or '无'}, "
            f"返回 {len(search_results)} 条结果"
        )

        return search_results

    def search_with_subject(
        self,
        query: str,
        subject: str,
        top_k: int = 10,
    ) -> List[VectorSearchResult]:
        """按科目过滤的向量检索"""
        return self.search(query, top_k, {"subject": subject})

    def search_with_file_type(
        self,
        query: str,
        file_type: str,
        top_k: int = 10,
    ) -> List[VectorSearchResult]:
        """按文件类型过滤的向量检索"""
        return self.search(query, top_k, {"file_type": file_type})

    @staticmethod
    def _build_filter_expr(
        metadata_filter: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        构建 Milvus 标量过滤表达式

        Args:
            metadata_filter: 过滤条件字典

        Returns:
            Milvus 表达式字符串 (如 'subject == "高数" && file_type == "pdf"')
        """
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} == {value}")
            elif isinstance(value, list):
                values_str = ", ".join(
                    f'"{v}"' if isinstance(v, str) else str(v) for v in value
                )
                conditions.append(f"{key} in [{values_str}]")

        return " && ".join(conditions) if conditions else None


# 全局向量检索引擎
vector_search_engine = VectorSearchEngine()
