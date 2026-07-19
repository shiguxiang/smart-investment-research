"""
综合研判 Agent
基于检索到的财务数据 + 用户问题，调用 Qwen-Max 生成结构化投研分析报告
支持 Chain-of-Thought 推理、跨公司对比、引用溯源
"""

import time
from typing import List, Dict, Optional
from http import HTTPStatus

import dashscope
from dashscope import Generation

from src.agents.state import AgentState
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisAgent:
    """
    综合研判 Agent
    职责: 接收财务检索结果 + 用户问题 → Qwen-Max 推理 → 生成带引用的投研分析
    核心能力: 跨年报财务对比、行业趋势研判、异常指标预警、引用溯源
    """

    # 系统提示词模板
    SYSTEM_PROMPT = """你是一个专业的智能投研分析助手。你的任务是基于上市公司年报数据，帮助分析师进行财务分析和投资研判。

## 你的能力
1. 基于提供的年报内容进行财务数据分析
2. 跨公司、跨行业对比关键财务指标（营收、净利润、ROE、毛利率等）
3. 识别财务数据中的异常波动和趋势变化
4. 引用具体的年报来源和段落

## 回答规则
1. **数据优先**: 所有分析结论必须基于提供的年报数据，不可凭空推测
2. **分步推理**: 先梳理数据 → 再对比分析 → 最后给出研判结论
3. **引用来源**: 每个关键数据点后标注引用编号，如 [1]、[2]，注明公司名称和年份
4. **口径说明**: 如果不同公司指标口径不一致，明确指出差异
5. **诚实表达**: 数据不足以支撑结论时，明确说明"基于现有数据无法确定"
6. **结构化输出**: 使用清晰的段落、表格或列表表达财务分析结果
7. **专业审慎**: 使用准确的财务术语，给出有数据支撑的投资参考建议

## 回答格式
```
[数据梳理]
(列示涉及的关键财务数据，含公司/年份/金额/同比变化)

[对比分析]
(跨公司或跨期对比，包含图表描述或趋势说明)

[综合研判]
(基于数据的投资分析结论和风险提示)

[引用来源]
[1] 公司名, 年报年份, 相关内容: ...
[2] 公司名, 年报年份, 相关内容: ...
```"""

    def __init__(self):
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout

        dashscope.api_key = settings.dashscope_api_key

        self.generation_params = {
            "temperature": 0.3,   # 低温度保证数据准确性
            "top_p": 0.8,
            "max_tokens": 2048,
            "result_format": "message",
        }

    def process(self, state: AgentState) -> AgentState:
        """
        执行综合研判任务

        输入: state.query, state.hybrid_results (财务检索上下文)
        输出: state.final_answer, state.references, state.reasoning_trace
        """
        logger.info(f"综合研判 Agent 开始处理: query='{state['query'][:80]}...'")

        state["current_step"] = "reasoning"

        try:
            # 1. 构建研判 Prompt
            messages = self._build_messages(state)

            # 2. 调用 Qwen-Max
            response = self._call_llm(messages)

            if response:
                # 3. 解析回答
                final_answer, references, reasoning_trace = self._parse_response(
                    response, state
                )

                state["final_answer"] = final_answer
                state["references"] = references
                state["reasoning_trace"] = reasoning_trace

                logger.info(
                    f"综合研判 Agent 完成: 回答长度 {len(final_answer)} 字符, "
                    f"引用 {len(references)} 条"
                )
            else:
                state["final_answer"] = "抱歉，投研分析生成失败，请稍后重试。"
                state["references"] = []
                state["reasoning_trace"] = ""
                state["error"] = "LLM 调用失败"

        except Exception as e:
            logger.error(f"综合研判 Agent 异常: {e}")
            state["final_answer"] = f"分析过程出现异常: {str(e)}"
            state["references"] = []
            state["reasoning_trace"] = ""
            state["error"] = str(e)

        return state

    def _build_messages(self, state: AgentState) -> List[Dict]:
        """构建完整对话消息"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # 添加对话历史 (最近3轮)
        history = state.get("history", [])
        for h in history[-6:]:
            messages.append(h)

        # 构建用户消息
        user_content = self._build_user_content(state)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, state: AgentState) -> str:
        """构建用户消息内容"""
        parts = []

        # 用户问题
        parts.append(f"## 用户分析需求\n{state['query']}")

        # 年报原文 (如果有)
        report_text = state.get("ocr_text", "")
        if report_text:
            truncated = report_text[:1500]
            if len(report_text) > 1500:
                truncated += "\n... (年报内容过长，已截断)"
            parts.append(f"## 上传年报内容\n{truncated}")

        # 检索到的财务数据
        hybrid_results = state.get("hybrid_results", [])
        if hybrid_results:
            parts.append("## 相关年报数据和财务指标\n")
            for i, doc in enumerate(hybrid_results[:8], 1):
                source_info = f"来源: {doc.get('file_name', '未知')}"
                if doc.get("subject"):
                    source_info += f", 行业/公司: {doc['subject']}"
                if doc.get("chapter"):
                    source_info += f", 年份: {doc['chapter']}"

                parts.append(
                    f"[{i}] {source_info}\n"
                    f"相关内容: {doc['text'][:500]}"
                )
                parts.append("")

        if not report_text and not hybrid_results:
            parts.append(
                "\n注意: 当前无可用的年报数据，请基于你的知识进行分析，"
                "并明确告知分析师当前分析未基于具体年报数据。"
            )

        return "\n".join(parts)

    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        """调用 Qwen-Max 生成分析"""
        for attempt in range(3):
            try:
                resp = Generation.call(
                    model=self.model,
                    messages=messages,
                    **self.generation_params,
                )

                if resp.status_code == HTTPStatus.OK:
                    return resp.output.choices[0].message.content
                else:
                    logger.warning(
                        f"Qwen-Max API 错误 (attempt {attempt + 1}): "
                        f"code={resp.status_code}, message={resp.message}"
                    )

            except Exception as e:
                logger.error(
                    f"Qwen-Max 调用异常 (attempt {attempt + 1}): {e}"
                )

            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

        return None

    def _parse_response(
        self, response: str, state: AgentState
    ) -> tuple:
        """
        解析 LLM 回答
        Returns: (final_answer, references, reasoning_trace)
        """
        reasoning_trace = ""
        references = []
        final_answer = response

        # 提取 [引用来源] 部分
        if "[引用来源]" in response:
            parts = response.split("[引用来源]")
            final_answer = parts[0].strip()
            ref_text = parts[1].strip() if len(parts) > 1 else ""

            for line in ref_text.split("\n"):
                line = line.strip()
                if line and line.startswith("[") and "]" in line[:5]:
                    references.append({"citation": line})

        # 提取 [数据梳理] 和 [综合研判] 部分
        if "[数据梳理]" in final_answer:
            analyze_split = final_answer.split("[数据梳理]")
            if len(analyze_split) > 1:
                remaining = analyze_split[1]
                if "[对比分析]" in remaining:
                    reasoning_trace = remaining.split("[对比分析]")[0].strip()
                elif "[综合研判]" in remaining:
                    reasoning_trace = remaining.split("[综合研判]")[0].strip()
                else:
                    reasoning_trace = remaining.strip()

        # 兜底: 基于检索结果构建引用
        if not references and state.get("hybrid_results"):
            for i, doc in enumerate(state["hybrid_results"][:5], 1):
                file_name = doc.get("file_name", "未知来源")
                snippet = doc["text"][:100]
                references.append({
                    "citation": f"[{i}] {file_name}: {snippet}..."
                })

        return final_answer, references, reasoning_trace
