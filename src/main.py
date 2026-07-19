"""
智能投研分析系统 — 服务入口
FastAPI 应用 + 生命周期管理

启动方式:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router
from src.retrieval.milvus_client import milvus_client
from src.retrieval.reranker import reranker
from src.utils.cache import cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时连接 Milvus + Redis, 关闭时释放资源
    """
    logger.info("=" * 60)
    logger.info(" 智能投研分析系统 启动中...")
    logger.info("=" * 60)

    # ===== Startup =====
    startup_errors = []

    # 1. 连接 Milvus
    try:
        milvus_client.connect()
        collection = milvus_client.create_collection()
        milvus_client.create_index(collection)
        milvus_client.load_collection(collection)
        logger.info(f"✓ Milvus 就绪 — Collection: {collection.name}")
    except Exception as e:
        startup_errors.append(f"Milvus: {e}")
        logger.warning(f"✗ Milvus 连接失败: {e}")

    # 2. 连接 Redis
    try:
        await cache.connect()
        logger.info("✓ Redis 就绪")
    except Exception as e:
        startup_errors.append(f"Redis: {e}")
        logger.warning(f"✗ Redis 连接失败: {e}")

    # 3. 预加载 Reranker (后台)
    try:
        reranker._ensure_model()
        logger.info(f"✓ Reranker 就绪 — {reranker.model_name}")
    except Exception as e:
        startup_errors.append(f"Reranker: {e}")
        logger.warning(f"✗ Reranker 加载失败: {e}")

    # 4. 总结
    if startup_errors:
        logger.warning(
            f"部分服务未就绪 ({len(startup_errors)}):\n"
            + "\n".join(f"  - {e}" for e in startup_errors)
        )
        logger.info("系统将以降级模式运行")
    else:
        logger.info("✓ 所有服务就绪")

    logger.info(f"API 地址: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"API 文档: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield  # 应用运行中

    # ===== Shutdown =====
    logger.info("应用关闭中...")
    milvus_client.disconnect()
    await cache.disconnect()
    logger.info("资源已释放，再见!")


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="智能投研分析系统",
    description=(
        "基于 Multi-Agent + RAG 的智能投研辅助系统\n\n"
        "**核心技术栈:** LangGraph · Milvus · Qwen-Max · BGE-Reranker · Ragas\n\n"
        "**核心能力:**\n"
        "- 📄 年报 PDF 智能解析 (版面分析 + 表格提取 + 跨页拼接)\n"
        "- 🔍 三路混合检索 (向量 + 关键词 + 行业/公司/年份过滤)\n"
        "- 🧠 多智能体协作分析 (文档解析Agent + 财务分析Agent + 综合研判Agent)\n"
        "- 📊 自动化质量评估 (Ragas: ContextPrecision + Faithfulness)\n"
        "- 🛡️ 工程化兜底 (表格解析降级 / LLM超时 / 缓存回复)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


# ==================== 根路径 ====================

@app.get("/")
async def root():
    """根路径 — 重定向到 API 文档"""
    return {
        "name": "智能投研分析系统",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat",
            "ingest": "/api/v1/ingest",
            "eval": "/api/v1/eval",
            "health": "/api/v1/health",
        },
    }


# ==================== 直接运行 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
