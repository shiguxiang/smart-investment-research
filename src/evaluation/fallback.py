"""
工程化兜底机制
保障服务可用性达 97%+

兜底策略:
1. OCR 识别失败 → 返回原始文件提示 + 建议手动输入
2. LLM 超时 → Redis 缓存的热门问答直接返回
3. Milvus 不可用 → 降级为纯关键词检索
4. 全局异常 → 友好错误提示 + 日志记录
"""

import time
import re
from typing import List, Dict, Optional

from src.agents.state import AgentState
from src.retrieval.keyword_search import bm25_index
from src.utils.cache import cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FallbackHandler:
    """
    兜底处理器
    每个策略独立可测，返回统一格式的 AgentState
    """

    # 缓存 TTL (热门问答缓存 24 小时)
    CACHE_TTL = 86400

    # LLM 超时时间 (秒)
    LLM_TIMEOUT = 30

    def handle(self, state: AgentState) -> AgentState:
        """
        主兜底入口
        根据错误类型自动选择最合适的降级策略
        """
        error = state.get("error", "")
        current_step = state.get("current_step", "")

        logger.info(f"触发兜底: step={current_step}, error={error}")

        # 策略 1: OCR 失败
        if current_step == "ocr_parsing" or "ocr" in error.lower():
            return self._handle_ocr_failure(state)

        # 策略 2: 检索失败 → 关键词降级
        if current_step == "retrieving" or "milvus" in error.lower():
            return self._handle_retrieval_failure(state)

        # 策略 3: LLM 超时 / API 不可用 → 缓存回答
        if current_step == "reasoning" or "timeout" in error.lower():
            return self._handle_llm_failure(state)

        # 策略 4: 全局兜底
        return self._handle_general_failure(state)

    def _handle_ocr_failure(self, state: AgentState) -> AgentState:
        """
        OCR 失败兜底
        策略: 提示用户手动输入文本内容或将文件转为另一种格式
        """
        files = state.get("files", [])
        file_names = [f.split("/")[-1].split("\\")[-1] for f in files]

        state["fallback_used"] = True
        state["current_step"] = "fallback_ocr"

        if state.get("query", "").strip():
            # 有文本查询 → 跳过 OCR 直接检索
            state["final_answer"] = (
                f"⚠️ 文件 '{', '.join(file_names)}' 解析失败，但已根据您的文字描述进行检索。\n"
                f"建议: 将文件转为清晰的 PDF 或直接粘贴题目文本以获得更准确的结果。"
            )
            # 仍然尝试检索
            state["ocr_text"] = ""
            state["ocr_success"] = False
        else:
            state["final_answer"] = (
                f"⚠️ 文件 '{', '.join(file_names)}' 解析失败。\n\n"
                f"可能原因:\n"
                f"1. 文件格式不支持或已损坏\n"
                f"2. 图片质量过低，文字无法识别\n"
                f"3. 文件为扫描版，字迹模糊\n\n"
                f"建议:\n"
                f"- 直接在此输入题目文本\n"
                f"- 将文件重新拍照/扫描后上传\n"
                f"- 尝试使用 PDF 格式代替图片"
            )

        return state

    def _handle_retrieval_failure(self, state: AgentState) -> AgentState:
        """
        检索失败兜底
        策略: 降级为纯 BM25 关键词检索 (不依赖 Milvus)
        """
        state["fallback_used"] = True
        state["current_step"] = "fallback_retrieval"

        query = state.get("query", "")

        if not query.strip():
            state["final_answer"] = "请提供具体的问题以便为您检索相关资料。"
            return state

        # 尝试 BM25 关键词检索
        try:
            results = bm25_index.search(query, top_k=5)

            if results:
                # 构建简化回答
                context_texts = [r.text[:300] for r in results[:3]]
                state["hybrid_results"] = [
                    {
                        "doc_id": r.doc_id,
                        "text": r.text,
                        "score": r.score,
                        "rank": i + 1,
                        "file_name": r.metadata.get("file_name", ""),
                    }
                    for i, r in enumerate(results)
                ]

                state["final_answer"] = (
                    f"⚠️ 向量检索服务暂时不可用，已降级为关键词匹配。\n\n"
                    f"根据您的查询，找到以下相关内容:\n\n"
                    + "\n---\n".join(context_texts)
                    + f"\n\n💡 如需更精准的语义匹配结果，请稍后重试。"
                )
                state["retrieval_success"] = True
            else:
                state["final_answer"] = (
                    "未找到与您问题相关的资料。\n\n"
                    "建议:\n"
                    "1. 尝试使用不同的关键词描述问题\n"
                    "2. 确认已上传相关行业的年报数据\n"
                    "3. 直接描述具体的知识点或题目"
                )
                state["retrieval_success"] = False

        except Exception as e:
            logger.error(f"关键词检索兜底也失败了: {e}")
            state["final_answer"] = (
                "检索服务暂时不可用，请稍后重试。\n"
                "建议直接查阅年报原文相关章节。"
            )
            state["retrieval_success"] = False

        return state

    def _handle_llm_failure(self, state: AgentState) -> AgentState:
        """
        LLM 失败兜底
        策略:
        1. 先查 Redis 缓存 → 热门问答直接返回
        2. 无缓存 → 返回检索结果原文 (不经过 LLM 加工)
        """
        state["fallback_used"] = True
        state["current_step"] = "fallback_llm"

        query = state.get("query", "")

        # Step 1: 查缓存
        cached = None
        try:
            import asyncio
            # 尝试在同步上下文中获取缓存
            cached = asyncio.get_event_loop().run_until_complete(
                cache.get_cached_answer(query, state.get("subject", ""))
            )
        except Exception:
            pass

        if cached:
            state["final_answer"] = (
                f"⚠️ AI 推理服务暂时不可用，以下是历史热门解答:\n\n{cached}"
            )
            return state

        # Step 2: 直接返回检索上下文 (不加工)
        hybrid_results = state.get("hybrid_results", [])
        if hybrid_results:
            contexts = [
                f"**[{i+1}] {r.get('file_name', '来源')}** (相关度: {r.get('score', 0):.2f})\n{r['text'][:500]}"
                for i, r in enumerate(hybrid_results[:3])
            ]

            state["final_answer"] = (
                f"⚠️ AI 推理服务暂时不可用（可能因请求高峰期或服务升级）。\n\n"
                f"以下是根据您的问题匹配到的相关资料原文，请参考:\n\n"
                + "\n\n---\n\n".join(contexts)
                + f"\n\n💡 AI 服务恢复正常后，将为您提供更精准的解答。"
            )
        else:
            state["final_answer"] = (
                "AI 推理服务暂时不可用，请稍后重试。\n"
                "如问题紧急，建议直接查阅年报原文或参考行业研报。"
            )

        return state

    def _handle_general_failure(self, state: AgentState) -> AgentState:
        """
        全局异常兜底
        策略: 友好的通用错误提示
        """
        error_msg = state.get("error", "未知错误")
        logger.error(f"全局兜底: {error_msg}")

        state["fallback_used"] = True
        state["current_step"] = "fallback_general"

        query = state.get("query", "")

        # 如果有查询内容，给出线索
        if query.strip():
            # 尝试用最简单的关键词匹配
            keywords = self._extract_keywords(query)
            if keywords:
                state["final_answer"] = (
                    f"抱歉，系统处理您的问题时遇到了问题。\n\n"
                    f"您的问题涉及: {', '.join(keywords[:5])}\n\n"
                    f"建议:\n"
                    f"1. 直接查阅年报中相关章节\n"
                    f"2. 尝试重新描述问题\n"
                    f"3. 参考同行业可比公司年报\n\n"
                    f"系统正在恢复中，请稍后重试。"
                )
                return state

        state["final_answer"] = (
            "抱歉，系统暂时遇到了技术问题。\n\n"
            "我们正在努力恢复服务，请您稍后重试。\n"
            "如有紧急疑问，建议查阅年报原文或参考行业研报。"
        )

        return state

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
        """提取关键词"""
        words = re.findall(r"[a-zA-Z0-9_一-龥㐀-䶿]{2,}", text)
        # 去重、按长度排序 (长的更可能是关键术语)
        unique_words = list(dict.fromkeys(words))
        unique_words.sort(key=len, reverse=True)
        return unique_words[:max_keywords]

    @staticmethod
    def get_cached_or_default(query: str, subject: str = "") -> Optional[str]:
        """
        同步获取缓存 (供 API 层在调用 LLM 前快速检查)
        """
        try:
            import asyncio
            return asyncio.get_event_loop().run_until_complete(
                cache.get_cached_answer(query, subject)
            )
        except Exception:
            return None

    @staticmethod
    def cache_answer_sync(query: str, answer: str, subject: str = ""):
        """同步缓存回答"""
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                cache.cache_answer(query, answer, subject)
            )
        except Exception:
            pass
