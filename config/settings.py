"""
全局配置模块
统一管理所有环境变量和服务参数
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """应用全局配置"""

    # === DashScope ===
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")

    # === Milvus ===
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection_name: str = Field(
        default="review_materials", alias="MILVUS_COLLECTION_NAME"
    )

    # === Redis ===
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # === 模型配置 ===
    llm_model: str = Field(default="qwen-max", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )

    # === 检索参数 ===
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_k: int = Field(default=20, alias="RETRIEVAL_CANDIDATE_K")
    vector_weight: float = Field(default=0.5, alias="VECTOR_WEIGHT")
    keyword_weight: float = Field(default=0.3, alias="KEYWORD_WEIGHT")
    metadata_weight: float = Field(default=0.2, alias="METADATA_WEIGHT")

    # === 文本分块 ===
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # === 服务配置 ===
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # === 评估阈值 ===
    context_precision_threshold: float = Field(default=0.75)
    faithfulness_threshold: float = Field(default=0.80)

    # === LLM 超时 (秒) ===
    llm_timeout: int = Field(default=30)
    ocr_timeout: int = Field(default=60)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


# 全局单例
settings = Settings()
