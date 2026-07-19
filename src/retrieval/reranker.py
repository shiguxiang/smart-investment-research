"""
BGE-Reranker 重排序模块
基于 FlagEmbedding 的 BAAI/bge-reranker-v2-m3 模型
对候选结果进行精细化成对重排序
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """重排序结果"""
    index: int  # 在原候选集中的索引
    doc_id: str
    text: str
    score: float  # 重排分数 [0, 1]
    metadata: Dict = field(default_factory=dict)


class Reranker:
    """
    BGE-Reranker 重排序器
    加载 BAAI/bge-reranker-v2-m3 进行文档-查询对打分
    """

    # 默认模型
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = None,
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: BGE Reranker 模型名称或路径
            use_fp16: 是否使用半精度推理 (节省显存)
            batch_size: 重排批处理大小
            max_length: 最大输入长度 (token数)
        """
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size
        self.max_length = max_length

        self._model = None  # 延迟加载
        self._loaded = False

    def _ensure_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import FlagReranker

            logger.info(f"加载 Reranker 模型: {self.model_name}")
            self._model = FlagReranker(
                self.model_name,
                use_fp16=True,
            )
            self._loaded = True
            logger.info(f"Reranker 模型加载完成: {self.model_name}")

        except ImportError:
            logger.error(
                "FlagEmbedding 未安装，无法加载 Reranker。"
                "请运行: pip install FlagEmbedding"
            )
            self._model = None
            self._loaded = False
        except Exception as e:
            logger.error(f"加载 Reranker 模型失败: {e}")
            self._model = None
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        """检查模型是否就绪"""
        return self._loaded

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, Dict]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """
        对候选文档重排序

        Args:
            query: 用户查询
            documents: [(text, metadata), ...] 候选文档列表
            top_k: 返回 Top-K 结果

        Returns:
            RerankResult 列表 (按分数降序)
        """
        self._ensure_model()

        if not self.is_ready:
            logger.warning("Reranker 不可用，返回原始顺序")
            return self._fallback_rerank(query, documents, top_k)

        if not documents:
            return []

        try:
            # 构建 [query, doc] 对
            pairs = [[query, doc[0][:self.max_length]] for doc in documents]

            # 模型打分
            scores = self._model.compute_score(pairs, normalize=True)

            # 处理单结果情况
            if isinstance(scores, float):
                scores = [scores]

            # 组装结果
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append(RerankResult(
                    index=i,
                    doc_id=doc[1].get("chunk_id", f"doc_{i}"),
                    text=doc[0],
                    score=float(score),
                    metadata=doc[1],
                ))

            # 按分数降序
            results.sort(key=lambda r: r.score, reverse=True)

            logger.debug(
                f"Reranker: {len(documents)} 候选 → Top {min(top_k, len(results))}, "
                f"最高分: {results[0].score:.4f}" if results else "Reranker: 无结果"
            )

            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranker 打分异常: {e}")
            return self._fallback_rerank(query, documents, top_k)

    def _fallback_rerank(
        self,
        query: str,
        documents: List[Tuple[str, Dict]],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """
        兜底重排: 使用简单的 TF-IDF 关键词匹配
        (当 BGE-Reranker 不可用时)
        """
        import re

        def simple_score(text: str, query: str) -> float:
            """简单的词重叠率得分"""
            query_words = set(re.findall(r"[\w一-鿿]+", query.lower()))
            text_words = set(re.findall(r"[a-zA-Z0-9_一-鿿㐀-䶿]+", text.lower()))
            if not query_words:
                return 0.5
            overlap = query_words & text_words
            return len(overlap) / len(query_words)

        results = []
        for i, (text, meta) in enumerate(documents):
            score = simple_score(text, query)
            results.append(RerankResult(
                index=i,
                doc_id=meta.get("chunk_id", f"doc_{i}"),
                text=text,
                score=score,
                metadata=meta,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# 全局 Reranker 实例
reranker = Reranker()
