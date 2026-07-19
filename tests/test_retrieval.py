"""
检索模块测试
"""

import pytest

from src.retrieval.keyword_search import BM25Index, KeywordSearchResult
from src.retrieval.reranker import Reranker


class TestBM25Index:
    """BM25 关键词检索测试"""

    @pytest.fixture
    def index(self):
        idx = BM25Index()
        docs = [
            {"id": "doc_1", "text": "矩阵的秩是非零子式的最高阶数", "metadata": {"subject": "线性代数"}},
            {"id": "doc_2", "text": "极限是函数在趋近某点时的逼近值", "metadata": {"subject": "高等数学"}},
            {"id": "doc_3", "text": "矩阵可以通过初等行变换求秩", "metadata": {"subject": "线性代数"}},
            {"id": "doc_4", "text": "导数是函数的变化率", "metadata": {"subject": "高等数学"}},
            {"id": "doc_5", "text": "TCP协议使用三次握手建立连接", "metadata": {"subject": "计算机网络"}},
        ]
        idx.add_documents(docs)
        return idx

    def test_build_index(self, index):
        assert index.doc_count == 5
        assert index.is_ready

    def test_search_relevant(self, index):
        results = index.search("矩阵的秩怎么求", top_k=3)
        assert len(results) > 0
        # doc_1 和 doc_3 应该排在前面
        assert results[0].doc_id in ("doc_1", "doc_3")

    def test_search_no_match(self, index):
        results = index.search("量子力学薛定谔方程", top_k=3)
        # 可能返回空或低分结果
        if results:
            assert results[0].score < 5  # 分数应该很低

    def test_metadata_filter(self, index):
        results = index.search(
            "矩阵的秩",
            top_k=5,
            metadata_filter={"subject": "线性代数"}
        )
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("subject") == "线性代数"

    def test_empty_query(self, index):
        results = index.search("", top_k=5)
        assert results == []


class TestReranker:
    """重排序测试"""

    def test_fallback_rerank(self):
        """测试兜底重排逻辑 (不需要模型加载)"""
        reranker = Reranker()

        # 不加载模型，直接使用 fallback
        query = "什么是矩阵的秩"
        documents = [
            ("矩阵的秩定义为非零子式的最高阶数", {"chunk_id": "1", "subject": "线性代数"}),
            ("TCP三次握手的过程包括SYN SYN-ACK ACK", {"chunk_id": "2", "subject": "计算机网络"}),
            ("导数是函数在某点的瞬时变化率", {"chunk_id": "3", "subject": "高等数学"}),
        ]

        results = reranker._fallback_rerank(query, documents, top_k=3)

        assert len(results) == 3
        # 矩阵相关的应该排第一
        assert "矩阵" in results[0].text
        # 分数应该降序
        assert results[0].score >= results[-1].score

    def test_fallback_empty(self):
        reranker = Reranker()
        results = reranker._fallback_rerank("test", [], top_k=5)
        assert results == []

    def test_fallback_single(self):
        reranker = Reranker()
        query = "极限的定义"
        documents = [
            ("极限是微积分的基础概念", {"chunk_id": "1"}),
        ]
        results = reranker._fallback_rerank(query, documents, top_k=5)
        assert len(results) == 1
        # 单结果时得分固定为1.0 (min==max)
        assert results[0].score >= 0

    def test_rerank_scores_are_normalized(self):
        """重排分数应该在 [0, 1] 范围内"""
        reranker = Reranker()
        query = "数学"
        documents = [
            ("2024年营收同比增长23%净利润15亿", {"chunk_id": "1"}),
            ("数学分析研究函数的性质", {"chunk_id": "2"}),
        ]
        results = reranker._fallback_rerank(query, documents, top_k=5)
        for r in results:
            assert 0 <= r.score <= 1


class TestHybridSearchWeightedFusion:
    """混合检索加权融合测试"""

    def test_normalize_scores(self):
        from src.retrieval.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine()
        pairs = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
        norm = engine._normalize_scores(pairs)

        assert norm["a"] == 1.0
        assert norm["c"] == 0.0
        assert 0 < norm["b"] < 1.0

    def test_normalize_single_score(self):
        from src.retrieval.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine()
        pairs = [("a", 5.0)]
        norm = engine._normalize_scores(pairs)

        assert norm["a"] == 1.0

    def test_normalize_empty(self):
        from src.retrieval.hybrid_search import HybridSearchEngine

        engine = HybridSearchEngine()
        norm = engine._normalize_scores([])
        assert norm == {}
