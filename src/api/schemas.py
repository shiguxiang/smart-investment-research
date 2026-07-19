"""
Pydantic 请求/响应模型
定义 API 数据契约
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import UploadFile


# ==================== 请求模型 ====================

class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(
        ...,
        description="用户问题",
        min_length=1,
        max_length=5000,
        examples=["请解释泰勒展开式的原理"],
    )
    subject: Optional[str] = Field(
        default="",
        description="科目 (用于过滤检索范围)",
        examples=["高等数学"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID (用于多轮对话)",
    )


class IngestRequest(BaseModel):
    """资料摄入请求 (元数据)"""
    subject: str = Field(
        ...,
        description="科目名称",
        examples=["线性代数"],
    )
    chapter: Optional[str] = Field(
        default="",
        description="章节",
    )
    semester: Optional[str] = Field(
        default="",
        description="学期 (如 2025-2026-1)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="标签 (如 ['期中', '重点'])",
    )


class EvalRequest(BaseModel):
    """评估请求"""
    test_set_path: Optional[str] = Field(
        default=None,
        description="测试集 JSON 文件路径",
    )
    subject: Optional[str] = Field(default="", description="按科目筛选评估范围")


# ==================== 响应模型 ====================

class DocReference(BaseModel):
    """文档引用"""
    doc_id: str
    text: str = Field(description="摘要文本 (前200字符)")
    score: float
    file_name: str = ""
    subject: str = ""
    chapter: str = ""


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(description="AI 生成的回答")
    references: List[DocReference] = Field(
        default_factory=list,
        description="引用的资料来源",
    )
    retrieved_docs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的相关文档",
    )
    fallback_used: bool = Field(
        default=False,
        description="是否使用了兜底策略",
    )
    session_id: str = Field(description="会话ID")
    processing_time_ms: float = Field(
        default=0.0,
        description="处理耗时 (毫秒)",
    )
    error: Optional[str] = Field(default=None, description="错误信息")


class IngestResponse(BaseModel):
    """摄入响应"""
    success: bool
    file_name: str
    file_type: str
    chunks_created: int = 0
    stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class BatchIngestResponse(BaseModel):
    """批量摄入响应"""
    total_files: int
    success_count: int
    fail_count: int
    results: List[IngestResponse] = Field(default_factory=list)


class EvalReportResponse(BaseModel):
    """评估报告响应"""
    timestamp: str
    total_samples: int
    avg_context_precision: float
    avg_faithfulness: float
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    pass_rate: float
    alerts: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str  # "healthy" | "degraded" | "unhealthy"
    milvus: str  # "connected" | "disconnected"
    redis: str   # "connected" | "disconnected"
    reranker: str  # "loaded" | "not_loaded"
    version: str = "1.0.0"
