"""
Ragas 评估流水线
自动化监控检索和生成质量:
- ContextPrecision: 检索上下文精确度
- Faithfulness: 回答忠实度
- ContextRecall: 上下文召回率
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """评估样本"""
    question: str
    answer: str
    contexts: List[str]  # 检索到的上下文
    ground_truth: str = ""  # 参考答案 (可选)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """单条评估结果"""
    question: str
    context_precision: float = 0.0
    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    passed: bool = True
    details: Dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """评估报告"""
    timestamp: str
    total_samples: int
    avg_context_precision: float = 0.0
    avg_faithfulness: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    pass_rate: float = 0.0
    results: List[EvalResult] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


class RagasPipeline:
    """
    Ragas 评估流水线
    支持在线评估 (实时) 和离线评估 (批量数据集)
    """

    def __init__(self):
        self._metrics = None  # 延迟加载 Ragas
        self._llm_available = False

        # 阈值
        self.precision_threshold = settings.context_precision_threshold
        self.faithfulness_threshold = settings.faithfulness_threshold

    def _ensure_ragas(self):
        """延迟初始化 Ragas (耗时的导入)"""
        if self._metrics is not None:
            return

        try:
            # 尝试导入 Ragas
            from ragas.metrics import (
                context_precision,
                faithfulness,
                context_recall,
                answer_relevancy,
            )
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.chat_models import ChatTongyi

            # 使用 Qwen 作为评估 LLM
            eval_llm = LangchainLLMWrapper(ChatTongyi(
                model_name=settings.llm_model,
                dashscope_api_key=settings.dashscope_api_key,
            ))

            self._metrics = {
                "context_precision": context_precision,
                "faithfulness": faithfulness,
                "context_recall": context_recall,
                "answer_relevancy": answer_relevancy,
            }
            self._llm = eval_llm
            self._llm_available = True

            logger.info("Ragas 评估框架初始化成功")

        except ImportError as e:
            logger.warning(f"Ragas 未安装或导入失败: {e}，将使用简化评估")
            self._metrics = {}
            self._llm_available = False
        except Exception as e:
            logger.error(f"Ragas 初始化异常: {e}")
            self._metrics = {}
            self._llm_available = False

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> EvalResult:
        """
        评估单条问答

        Args:
            question: 用户问题
            answer: 系统生成的回答
            contexts: 检索到的上下文列表
            ground_truth: 参考答案 (可选)

        Returns:
            EvalResult 包含各项指标
        """
        self._ensure_ragas()

        result = EvalResult(question=question)

        if not self._llm_available or not contexts:
            # 简化评估: 基于规则的快速检查
            cp = self._simple_context_precision(question, contexts)
            ff = self._simple_faithfulness(answer, contexts)
            result.context_precision = cp
            result.faithfulness = ff
            result.passed = cp >= self.precision_threshold and ff >= self.faithfulness_threshold
            result.details = {"method": "heuristic"}
            return result

        # 使用 Ragas 完整评估
        try:
            from ragas import evaluate, EvaluationDataset
            from ragas.metrics import (
                context_precision,
                faithfulness,
                context_recall,
                answer_relevancy,
            )

            # 构建评估数据集
            dataset = EvaluationDataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth or answer],
            })

            # 执行评估
            scores = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    faithfulness,
                    context_recall,
                    answer_relevancy,
                ],
                llm=self._llm,
            )

            # 提取分数
            result.context_precision = float(scores.get("context_precision", 0))
            result.faithfulness = float(scores.get("faithfulness", 0))
            result.context_recall = float(scores.get("context_recall", 0))
            result.answer_relevancy = float(scores.get("answer_relevancy", 0))

            result.passed = (
                result.context_precision >= self.precision_threshold
                and result.faithfulness >= self.faithfulness_threshold
            )
            result.details = {"method": "ragas", "scores": scores}

        except Exception as e:
            logger.error(f"Ragas 评估异常: {e}")
            # 降级为启发式评估
            result.context_precision = self._simple_context_precision(question, contexts)
            result.faithfulness = self._simple_faithfulness(answer, contexts)
            result.passed = (
                result.context_precision >= self.precision_threshold
                and result.faithfulness >= self.faithfulness_threshold
            )
            result.details = {"method": "fallback", "error": str(e)}

        return result

    def evaluate_batch(
        self,
        samples: List[EvalSample],
    ) -> EvalReport:
        """
        批量评估

        Args:
            samples: 评估样本列表

        Returns:
            EvalReport 汇总报告
        """
        logger.info(f"开始批量评估: {len(samples)} 条样本")

        report = EvalReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(samples),
        )

        for sample in samples:
            result = self.evaluate_single(
                question=sample.question,
                answer=sample.answer,
                contexts=sample.contexts,
                ground_truth=sample.ground_truth,
            )
            report.results.append(result)

        # 计算平均值
        n = len(report.results)
        if n > 0:
            report.avg_context_precision = sum(
                r.context_precision for r in report.results
            ) / n
            report.avg_faithfulness = sum(
                r.faithfulness for r in report.results
            ) / n
            report.avg_context_recall = sum(
                r.context_recall for r in report.results
            ) / n
            report.avg_answer_relevancy = sum(
                r.answer_relevancy for r in report.results
            ) / n
            report.pass_rate = sum(
                1 for r in report.results if r.passed
            ) / n

        # 生成告警
        report.alerts = self._generate_alerts(report)

        logger.info(
            f"评估完成: ContextPrecision={report.avg_context_precision:.3f}, "
            f"Faithfulness={report.avg_faithfulness:.3f}, "
            f"PassRate={report.pass_rate:.1%}, "
            f"告警 {len(report.alerts)} 条"
        )

        return report

    def _generate_alerts(self, report: EvalReport) -> List[str]:
        """生成评估告警"""
        alerts = []

        if report.avg_context_precision < self.precision_threshold:
            alerts.append(
                f"⚠️ ContextPrecision={report.avg_context_precision:.2f} "
                f"低于阈值 {self.precision_threshold}，检索精度需优化"
            )

        if report.avg_faithfulness < self.faithfulness_threshold:
            alerts.append(
                f"⚠️ Faithfulness={report.avg_faithfulness:.2f} "
                f"低于阈值 {self.faithfulness_threshold}，生成幻觉率偏高"
            )

        if report.pass_rate < 0.7:
            alerts.append(
                f"🔴 整体通过率仅 {report.pass_rate:.1%}，建议排查检索和生成链路"
            )

        # 检查低分样本
        low_score_count = sum(
            1 for r in report.results
            if r.context_precision < 0.3 or r.faithfulness < 0.3
        )
        if low_score_count > len(report.results) * 0.2:
            alerts.append(
                f"⚠️ 低分样本占比 {low_score_count / len(report.results):.1%}"
            )

        return alerts

    # ==================== 启发式评估 (无 Ragas 时的兜底) ====================

    @staticmethod
    def _simple_context_precision(
        question: str, contexts: List[str]
    ) -> float:
        """
        简化上下文精度评估
        基于关键词重叠率判断上下文与问题的相关性
        """
        import re

        def extract_keywords(text: str) -> set:
            words = re.findall('[一-龥a-zA-Z0-9_]{2}', text.lower())
            return set(words)

        if not contexts:
            return 0.0

        q_keywords = extract_keywords(question)
        if not q_keywords:
            return 0.5

        scores = []
        for ctx in contexts:
            ctx_keywords = extract_keywords(ctx)
            if not ctx_keywords:
                scores.append(0.0)
            else:
                overlap = q_keywords & ctx_keywords
                scores.append(len(overlap) / len(q_keywords))

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _simple_faithfulness(answer: str, contexts: List[str]) -> float:
        """
        简化忠实度评估
        检查回答中的关键断言是否在上下文中出现
        """
        import re

        if not answer or not contexts:
            return 0.0

        # 提取回答中的句子
        sentences = re.split(r"[。！？!?\n]", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 0.5

        combined_ctx = " ".join(contexts)

        supported = 0
        for sent in sentences:
            # 检查句子中的关键词是否在上下文中
            words = re.findall(r"[一-龥a-zA-Z0-9_]{2,}", sent)
            if not words:
                supported += 0.5
                continue

            matched = sum(1 for w in words if w in combined_ctx)
            if matched / len(words) > 0.3:  # 30% 关键词匹配视为有依据
                supported += 1

        return supported / len(sentences)

    # ==================== 报告导出 ====================

    def save_report(self, report: EvalReport, file_path: str):
        """保存评估报告到 JSON 文件"""
        data = {
            "timestamp": report.timestamp,
            "total_samples": report.total_samples,
            "avg_context_precision": report.avg_context_precision,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_context_recall": report.avg_context_recall,
            "avg_answer_relevancy": report.avg_answer_relevancy,
            "pass_rate": report.pass_rate,
            "alerts": report.alerts,
            "results": [
                {
                    "question": r.question[:100],
                    "context_precision": r.context_precision,
                    "faithfulness": r.faithfulness,
                    "context_recall": r.context_recall,
                    "answer_relevancy": r.answer_relevancy,
                    "passed": r.passed,
                }
                for r in report.results
            ],
        }

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已保存: {file_path}")


# 全局评估流水线
ragas_pipeline = RagasPipeline()
