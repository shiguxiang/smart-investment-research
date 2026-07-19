"""
向量化模块
基于 DashScope text-embedding-v3 API 进行文本向量化
支持批量处理，含重试与超时机制
"""

import time
from typing import List, Optional
from http import HTTPStatus

import dashscope
from dashscope import TextEmbedding

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    文本向量化器
    封装 DashScope Embedding API (text-embedding-v3, 1024维)
    """

    # DashScope Embedding 模型参数
    MODEL_NAME = "text-embedding-v3"
    DIMENSION = 1024
    MAX_BATCH_SIZE = 25  # API 限制每批次最多 25 条
    MAX_TEXT_LENGTH = 6000  # 单条文本最大字符数

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key: DashScope API Key (不传则从配置读取)
            max_retries: 最大重试次数
            retry_delay: 重试间隔 (秒)
        """
        self.api_key = api_key or settings.dashscope_api_key
        dashscope.api_key = self.api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed(self, text: str) -> Optional[List[float]]:
        """
        单条文本向量化

        Args:
            text: 输入文本

        Returns:
            1024 维向量，失败返回 None
        """
        if not text or not text.strip():
            logger.warning("向量化输入为空")
            return None

        # 截断过长文本
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH]
            logger.debug(f"文本过长，已截断至 {self.MAX_TEXT_LENGTH} 字符")

        for attempt in range(self.max_retries):
            try:
                resp = TextEmbedding.call(
                    model=self.MODEL_NAME,
                    input=text,
                )

                if resp.status_code == HTTPStatus.OK:
                    embedding = resp.output["embeddings"][0]["embedding"]
                    return embedding
                else:
                    logger.warning(
                        f"Embedding API 返回错误 (attempt {attempt + 1}): "
                        f"code={resp.status_code}, message={resp.message}"
                    )
            except Exception as e:
                logger.error(f"Embedding API 调用异常 (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # 递增退避

        logger.error(f"向量化失败，已重试 {self.max_retries} 次")
        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表

        Returns:
            向量列表 (与输入一一对应，失败项为 None)
        """
        if not texts:
            return []

        results = []
        total = len(texts)

        # 分批调用 (API 限制每批次最多 25 条)
        for i in range(0, total, self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            batch_results = self._embed_single_batch(batch)
            results.extend(batch_results)

            if i + self.MAX_BATCH_SIZE < total:
                time.sleep(0.2)  # 批次间短暂间隔，避免限流

        return results

    def _embed_single_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """单批次向量化"""
        valid_texts = []
        valid_indices = []
        results = [None] * len(texts)

        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(
                    text[: self.MAX_TEXT_LENGTH] if len(text) > self.MAX_TEXT_LENGTH else text
                )
                valid_indices.append(idx)

        if not valid_texts:
            return results

        for attempt in range(self.max_retries):
            try:
                resp = TextEmbedding.call(
                    model=self.MODEL_NAME,
                    input=valid_texts,
                )

                if resp.status_code == HTTPStatus.OK:
                    embeddings = resp.output["embeddings"]
                    for i, emb_item in enumerate(embeddings):
                        original_idx = valid_indices[i]
                        results[original_idx] = emb_item["embedding"]
                    return results
                else:
                    logger.warning(
                        f"Batch Embedding API 返回错误 (attempt {attempt + 1}): "
                        f"code={resp.status_code}, message={resp.message}"
                    )
            except Exception as e:
                logger.error(f"Batch Embedding 异常 (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        logger.error(f"Batch 向量化失败，已重试 {self.max_retries} 次")
        return results

    def embed_with_cache(
        self,
        text: str,
        cache_dict: dict,
    ) -> Optional[List[float]]:
        """
        带本地缓存的向量化 (避免重复计算)

        Args:
            text: 输入文本
            cache_dict: 缓存字典 (text → embedding)

        Returns:
            1024 维向量
        """
        import hashlib

        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in cache_dict:
            return cache_dict[key]

        embedding = self.embed(text)
        if embedding:
            cache_dict[key] = embedding

        return embedding
