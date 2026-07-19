"""
API 路由定义
RESTful 接口: 对话、数据摄入、评估、健康检查
"""

import os
import time
import tempfile
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocReference,
    IngestRequest,
    IngestResponse,
    BatchIngestResponse,
    EvalRequest,
    EvalReportResponse,
    HealthResponse,
)
from src.agents.orchestrator import orchestrator
from src.ingestion.pipeline import IngestionPipeline, IngestResult
from src.evaluation.ragas_pipeline import ragas_pipeline, EvalSample, EvalReport
from src.evaluation.fallback import FallbackHandler
from src.retrieval.milvus_client import milvus_client
from src.retrieval.reranker import reranker
from src.utils.cache import cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["智能投研分析"])

# 初始化组件
ingestion_pipeline = IngestionPipeline()
fallback_handler = FallbackHandler()

# 上传文件保存目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==================== 对话接口 ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: str = Form(..., description="用户问题"),
    subject: str = Form("", description="科目"),
    session_id: str = Form("", description="会话ID"),
    files: List[UploadFile] = File(default=[], description="附件 (PDF/PPT/图片)"),
):
    """
    智能投研分析接口

    支持:
    - 纯文本分析需求 → 财务检索 + 综合研判
    - 带年报附件 → 文档解析 + 财务检索 + 综合研判

    返回: 分析报告 + 引用来源
    """
    start_time = time.perf_counter()

    # 1. 保存上传文件
    saved_files = []
    for upload_file in files:
        if upload_file.filename:
            file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
            content = await upload_file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file_path)

    # 2. 执行多智能体问答
    try:
        result = orchestrator.run_sync(
            query=query,
            files=saved_files if saved_files else None,
            subject=subject,
            session_id=session_id or None,
        )
    except Exception as e:
        logger.error(f"问答执行异常: {e}")
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")

    # 3. 构建响应
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    references = []
    for ref in result.get("references", []):
        references.append(DocReference(
            doc_id=ref.get("doc_id", ""),
            text=ref.get("citation", ref.get("text", ""))[:200],
            score=ref.get("score", 0.0),
            file_name=ref.get("file_name", ""),
            subject=ref.get("subject", ""),
            chapter=ref.get("chapter", ""),
        ))

    return ChatResponse(
        answer=result.get("answer", ""),
        references=references,
        retrieved_docs=result.get("retrieved_docs", []),
        fallback_used=result.get("fallback_used", False),
        session_id=result.get("session_id", session_id or ""),
        processing_time_ms=round(elapsed_ms, 2),
        error=result.get("error"),
    )


# ==================== 资料摄入接口 ====================

@router.post("/ingest", response_model=BatchIngestResponse)
async def ingest_materials(
    files: List[UploadFile] = File(..., description="资料文件 (PDF/PPT/图片)"),
    subject: str = Form(..., description="科目"),
    chapter: str = Form("", description="章节"),
):
    """
    年报资料摄入接口

    上传上市公司年报 PDF，自动解析并存入向量数据库
    支持版面分析、表格提取、跨页拼接
    支持批量上传
    """
    results = []
    success_count = 0
    fail_count = 0

    for upload_file in files:
        if not upload_file.filename:
            continue

        # 保存文件
        file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
        content = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 摄入处理
        metadata = {
            "subject": subject,
            "chapter": chapter,
        }

        try:
            ingest_result = ingestion_pipeline.ingest_file(file_path, metadata)

            if ingest_result.success:
                success_count += 1
            else:
                fail_count += 1

            results.append(IngestResponse(
                success=ingest_result.success,
                file_name=ingest_result.file_name,
                file_type=ingest_result.file_type,
                chunks_created=len(ingest_result.chunks),
                stats=ingest_result.stats,
                error=ingest_result.error,
            ))

        except Exception as e:
            fail_count += 1
            results.append(IngestResponse(
                success=False,
                file_name=upload_file.filename,
                file_type="unknown",
                error=str(e),
            ))

    return BatchIngestResponse(
        total_files=len(files),
        success_count=success_count,
        fail_count=fail_count,
        results=results,
    )


@router.post("/ingest/single", response_model=IngestResponse)
async def ingest_single_file(
    file: UploadFile = File(...),
    subject: str = Form(...),
    chapter: str = Form(""),
):
    """单文件摄入"""
    # 保存
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 处理
    metadata = {"subject": subject, "chapter": chapter}
    result = ingestion_pipeline.ingest_file(file_path, metadata)

    return IngestResponse(
        success=result.success,
        file_name=result.file_name,
        file_type=result.file_type,
        chunks_created=len(result.chunks),
        stats=result.stats,
        error=result.error,
    )


# ==================== 评估接口 ====================

@router.get("/eval", response_model=EvalReportResponse)
async def run_evaluation(
    background_tasks: BackgroundTasks,
    subject: str = "",
):
    """
    触发评估任务

    运行 Ragas 评估流水线，检查 ContextPrecision 和 Faithfulness
    """
    logger.info(f"触发评估: subject={subject or '全部'}")

    try:
        # 加载测试集 (简化: 使用内置样例)
        samples = _load_test_samples(subject)

        if not samples:
            return EvalReportResponse(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_samples=0,
                avg_context_precision=0,
                avg_faithfulness=0,
                pass_rate=0,
                alerts=["测试集为空，请先创建评估样本"],
            )

        # 执行评估
        eval_samples = []
        for s in samples:
            eval_samples.append(EvalSample(
                question=s["question"],
                answer=s.get("answer", ""),
                contexts=s.get("contexts", []),
                ground_truth=s.get("ground_truth", ""),
                metadata=s.get("metadata", {}),
            ))

        report = ragas_pipeline.evaluate_batch(eval_samples)

        # 后台保存报告
        report_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "eval_reports",
            f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        ragas_pipeline.save_report(report, report_path)

        return EvalReportResponse(
            timestamp=report.timestamp,
            total_samples=report.total_samples,
            avg_context_precision=round(report.avg_context_precision, 4),
            avg_faithfulness=round(report.avg_faithfulness, 4),
            avg_context_recall=round(report.avg_context_recall, 4),
            avg_answer_relevancy=round(report.avg_answer_relevancy, 4),
            pass_rate=round(report.pass_rate, 4),
            alerts=report.alerts,
        )

    except Exception as e:
        logger.error(f"评估执行异常: {e}")
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


def _load_test_samples(subject: str = "") -> List[dict]:
    """加载测试样本 (从内置样例或文件)"""
    # 内置样例
    builtin_samples = [
        {
            "question": "什么是矩阵的秩？",
            "answer": "矩阵的秩是其非零子式的最高阶数，也是行向量或列向量组的极大无关组所含向量的个数。",
            "contexts": [
                "矩阵的秩定义为矩阵中不等于0的子式的最高阶数。矩阵A的秩记作r(A)或rank(A)。秩也是矩阵行向量组（或列向量组）的极大无关组所含向量的个数。",
            ],
            "ground_truth": "矩阵的秩是非零子式的最高阶数，等于行秩等于列秩。",
            "metadata": {"subject": "线性代数"},
        },
        {
            "question": "泰勒展开式的一般形式是什么？",
            "answer": "f(x) = f(x₀) + f'(x₀)(x-x₀) + f''(x₀)(x-x₀)²/2! + ... + fⁿ(x₀)(x-x₀)ⁿ/n! + Rₙ(x)",
            "contexts": [
                "泰勒中值定理：若函数f(x)在x₀的某邻域内n+1阶可导，则f(x)=f(x₀)+f'(x₀)(x-x₀)+...+fⁿ(x₀)/n!(x-x₀)ⁿ+Rₙ(x)，其中Rₙ(x)为余项。",
            ],
            "ground_truth": "f(x)=∑f⁽ⁿ⁾(x₀)/n!·(x-x₀)ⁿ",
            "metadata": {"subject": "高等数学"},
        },
    ]

    if subject:
        return [s for s in builtin_samples if s["metadata"].get("subject") == subject]
    return builtin_samples


# ==================== 健康检查接口 ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    检查 Milvus、Redis、Reranker 等服务状态
    """
    # Milvus
    milvus_status = "disconnected"
    try:
        if milvus_client.is_connected:
            milvus_status = "connected"
    except Exception:
        pass

    # Redis
    redis_status = "disconnected"
    try:
        if cache.client:
            await cache.client.ping()
            redis_status = "connected"
    except Exception:
        pass

    # Reranker
    reranker_status = "loaded" if reranker.is_ready else "not_loaded"

    # 综合状态
    if milvus_status == "connected" and redis_status == "connected":
        status = "healthy"
    elif milvus_status == "connected" or redis_status == "connected":
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        milvus=milvus_status,
        redis=redis_status,
        reranker=reranker_status,
    )
