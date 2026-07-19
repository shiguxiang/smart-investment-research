"""
混合检索模块 — 三路召回 + 加权融合
1. 关键词召回 (BM25)
2. 向量召回 (Milvus)
3. 元数据过滤
融合后交给 BGE-Reranker 精排
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from src.retrieval.keyword_search import BM25Index, KeywordSearchResult, bm25_index
from src.retrieval.vector_search import VectorSearchResult, vector_search_engine
from src.retrieval.reranker import Reranker, reranker, RerankResult
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    """混合检索最终结果"""
    doc_id: str
    text: str
    score: float  # 融合 + 重排后的最终分数
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    rank: int = 0  # 最终排名


class HybridSearchEngine:
    """
    混合检索引擎
    三路召回 → 加权融合 → BGE-Reranker 精排
    """

    def __init__(
        self,
        vector_weight: float = None,
        keyword_weight: float = None,
        metadata_weight: float = None,
    ):
        """
        Args:
            vector_weight: 向量召回权重
            keyword_weight: 关键词召回权重
            metadata_weight: 元数据匹配权重
        """
        self.vector_weight = vector_weight or settings.vector_weight
        self.keyword_weight = keyword_weight or settings.keyword_weight
        self.metadata_weight = metadata_weight or settings.metadata_weight

        self._reranker = Reranker()

    def search(
        self,
        query: str,
        top_k: int = None,
        candidate_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        """
        执行混合检索

        Args:
            query: 用户查询
            top_k: 最终返回结果数 (默认5)
            candidate_k: 候选集大小 (值越大精度越高，速度越慢)
            metadata_filter: 元数据过滤条件

        Returns:
            HybridSearchResult 列表 (Top-K)
        """
        top_k = top_k or settings.retrieval_top_k
        candidate_k = candidate_k or settings.retrieval_candidate_k

        logger.info(f"混合检索: query='{query[:80]}...', top_k={top_k}, candidate_k={candidate_k}")

        # ========== 第1路: 向量召回 ==========
        vector_results = vector_search_engine.search(
            query=query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        # ========== 第2路: 关键词召回 ==========
        keyword_results = bm25_index.search(
            query=query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        # ========== 第3路: 元数据匹配 (文件名/科目) ==========
        metadata_results = []
        if not metadata_filter and bm25_index.is_ready:
            metadata_results = bm25_index.search_all_fields(
                query=query,
                top_k=candidate_k // 2,
            )

        # ========== 加权融合 ==========
        fused = self._weighted_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            metadata_results=metadata_results,
            candidate_k=candidate_k,
        )

        logger.info(
            f"三路召回: 向量 {len(vector_results)}, "
            f"关键词 {len(keyword_results)}, "
            f"元数据 {len(metadata_results)} → 融合 {len(fused)} 候选"
        )

        # ========== BGE-Reranker 精排 ==========
        if fused:
            ranked = self._reranker.rerank(
                query=query,
                documents=[(item["text"], item["metadata"]) for item in fused],
                top_k=min(top_k, len(fused)),
            )

            # 组装最终结果
            final_results = []
            for rank, rr in enumerate(ranked):
                # 找到原始融合数据
                fused_item = fused[rr.index] if rr.index < len(fused) else None
                final_results.append(HybridSearchResult(
                    doc_id=fused_item["doc_id"] if fused_item else rr.doc_id,
                    text=rr.text,
                    score=rr.score,
                    vector_score=fused_item.get("vector_score", 0) if fused_item else 0,
                    keyword_score=fused_item.get("keyword_score", 0) if fused_item else 0,
                    rerank_score=rr.score,
                    metadata=fused_item["metadata"] if fused_item else {},
                    rank=rank + 1,
                ))
        else:
            final_results = []

        logger.info(f"混合检索完成: 返回 {len(final_results)} 条结果")
        return final_results

    def search_simple(
        self, query: str, top_k: int = 5, subject: str = ""
    ) -> List[HybridSearchResult]:
        """
        简化接口 — 快速检索

        Args:
            query: 用户查询
            top_k: 返回数量
            subject: 科目过滤 (可选)

        Returns:
            HybridSearchResult 列表
        """
        metadata_filter = {"subject": subject} if subject else None
        return self.search(query, top_k=top_k, metadata_filter=metadata_filter)

    def _weighted_fusion(
        self,
        vector_results: List[VectorSearchResult],
        keyword_results: List[KeywordSearchResult],
        metadata_results: List[KeywordSearchResult],
        candidate_k: int,
    ) -> List[Dict]:
        """
        加权融合三路召回结果

        算法:
        1. 对每路结果的分数做 min-max 归一化
        2. 按权重加权求和
        3. 去重合并，取 Top-candidate_k
        """

        # Step 1: 归一化各路线分数
        vector_norm = self._normalize_scores(
            [(r.doc_id, r.score) for r in vector_results]
        )
        keyword_norm = self._normalize_scores(
            [(r.doc_id, r.score) for r in keyword_results]
        )
        metadata_norm = self._normalize_scores(
            [(r.doc_id, r.score) for r in metadata_results]
        )

        # Step 2: 加权融合
        fused_map: Dict[str, Dict] = {}

        # 向量召回
        for doc_id, norm_score in vector_norm.items():
            if doc_id not in fused_map:
                fused_map[doc_id] = {
                    "doc_id": doc_id,
                    "text": "",
                    "metadata": {},
                    "vector_score": norm_score,
                    "keyword_score": 0.0,
                    "metadata_score": 0.0,
                    "fused_score": 0.0,
                }
            fused_map[doc_id]["vector_score"] = norm_score

        # 填充向量结果详情
        for vr in vector_results:
            if vr.doc_id in fused_map and not fused_map[vr.doc_id]["text"]:
                fused_map[vr.doc_id]["text"] = vr.text
                fused_map[vr.doc_id]["metadata"] = vr.metadata

        # 关键词召回
        for doc_id, norm_score in keyword_norm.items():
            if doc_id not in fused_map:
                fused_map[doc_id] = {
                    "doc_id": doc_id,
                    "text": "",
                    "metadata": {},
                    "vector_score": 0.0,
                    "keyword_score": norm_score,
                    "metadata_score": 0.0,
                    "fused_score": 0.0,
                }
            else:
                fused_map[doc_id]["keyword_score"] = norm_score

        # 填充关键词结果详情
        for kr in keyword_results:
            if kr.doc_id in fused_map and not fused_map[kr.doc_id]["text"]:
                fused_map[kr.doc_id]["text"] = kr.text
                fused_map[kr.doc_id]["metadata"] = kr.metadata

        # 元数据匹配
        for doc_id, norm_score in metadata_norm.items():
            if doc_id not in fused_map:
                fused_map[doc_id] = {
                    "doc_id": doc_id,
                    "text": "",
                    "metadata": {},
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                    "metadata_score": norm_score,
                    "fused_score": 0.0,
                }
            else:
                fused_map[doc_id]["metadata_score"] = norm_score

        # Step 3: 计算加权融合分数
        for doc_id, item in fused_map.items():
            item["fused_score"] = (
                self.vector_weight * item["vector_score"]
                + self.keyword_weight * item["keyword_score"]
                + self.metadata_weight * item["metadata_score"]
            )

        # Step 4: 排序取 Top-candidate_k
        sorted_items = sorted(
            fused_map.values(),
            key=lambda x: x["fused_score"],
            reverse=True,
        )

        return sorted_items[:candidate_k]

    @staticmethod
    def _normalize_scores(
        id_score_pairs: List[tuple],
    ) -> Dict[str, float]:
        """
        Min-Max 归一化分数到 [0, 1]

        Args:
            id_score_pairs: [(doc_id, score), ...]

        Returns:
            {doc_id: normalized_score}
        """
        if not id_score_pairs:
            return {}

        scores = [s for _, s in id_score_pairs]
        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return {doc_id: 1.0 for doc_id, _ in id_score_pairs}

        return {
            doc_id: (score - min_s) / (max_s - min_s)
            for doc_id, score in id_score_pairs
        }


# 全局混合检索引擎
hybrid_search_engine = HybridSearchEngine()
