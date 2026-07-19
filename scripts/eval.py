#!/usr/bin/env python3
"""
Ragas 评估执行脚本
运行自动化评估流水线，生成质量报告

用法:
    python scripts/eval.py --test-set data/test_set.json
    python scripts/eval.py --quick  # 快速评估 (内置样例)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.ragas_pipeline import (
    RagasPipeline,
    EvalSample,
    EvalReport,
)
from src.agents.orchestrator import orchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ragas 评估工具")

    parser.add_argument(
        "--test-set",
        type=str,
        default="",
        help="测试集 JSON 文件路径",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用内置样例快速评估",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="按科目过滤评估范围",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="评估报告输出路径",
    )
    parser.add_argument(
        "--run-qa",
        action="store_true",
        help="运行完整问答流程生成评估数据 (较慢)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" Ragas 评估流水线")
    print("=" * 60)

    # 初始化
    pipeline = RagasPipeline()

    # 加载测试集
    if args.test_set:
        print(f"\n加载测试集: {args.test_set}")
        samples = load_test_set(args.test_set)
    elif args.quick:
        print("\n使用内置样例...")
        samples = get_builtin_samples()
    else:
        print("\n使用内置样例 (--quick)...")
        samples = get_builtin_samples()

    if not samples:
        print("❌ 测试集为空")
        sys.exit(1)

    print(f"✓ 测试集加载: {len(samples)} 条样本")

    # 如果指定 --run-qa，先运行问答生成回答
    if args.run_qa:
        print(f"\n运行完整问答流程 ({len(samples)} 条)...")
        samples = run_qa_for_samples(samples)

    # 过滤科目
    if args.subject:
        samples = [s for s in samples
                   if s.get("metadata", {}).get("subject") == args.subject]
        print(f"  过滤后: {len(samples)} 条 (科目: {args.subject})")

    # 执行评估
    print(f"\n执行 Ragas 评估...")
    start = time.time()

    eval_samples = []
    for s in samples:
        eval_samples.append(EvalSample(
            question=s["question"],
            answer=s.get("answer", ""),
            contexts=s.get("contexts", []),
            ground_truth=s.get("ground_truth", ""),
            metadata=s.get("metadata", {}),
        ))

    report = pipeline.evaluate_batch(eval_samples)

    elapsed = time.time() - start
    print(f"评估耗时: {elapsed:.1f}s")

    # 打印报告
    print("\n" + "=" * 60)
    print(" 📊 评估报告")
    print("=" * 60)
    print(f"时间: {report.timestamp}")
    print(f"样本数: {report.total_samples}")
    print(f"-" * 40)
    print(f"ContextPrecision:   {report.avg_context_precision:.4f}  "
          f"{'✓' if report.avg_context_precision >= 0.75 else '⚠️'}")
    print(f"Faithfulness:        {report.avg_faithfulness:.4f}  "
          f"{'✓' if report.avg_faithfulness >= 0.80 else '⚠️'}")
    print(f"ContextRecall:       {report.avg_context_recall:.4f}")
    print(f"AnswerRelevancy:     {report.avg_answer_relevancy:.4f}")
    print(f"-" * 40)
    print(f"通过率: {report.pass_rate:.1%}")
    print(f"告警: {len(report.alerts)} 条")

    if report.alerts:
        print(f"\n⚠️ 告警信息:")
        for alert in report.alerts:
            print(f"  {alert}")

    # 打印各样本详情
    print(f"\n📋 样本详情:")
    for i, r in enumerate(report.results):
        status = "✓" if r.passed else "✗"
        print(f"  [{status}] 样本{i+1}: CP={r.context_precision:.3f}, "
              f"FF={r.faithfulness:.3f}, CR={r.context_recall:.3f}")

    # 保存报告
    output_path = args.output
    if not output_path:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "eval_reports"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

    pipeline.save_report(report, output_path)
    print(f"\n✓ 报告已保存: {output_path}")

    return report


def load_test_set(file_path: str) -> list:
    """从 JSON 文件加载测试集"""
    if not os.path.exists(file_path):
        print(f"❌ 测试集文件不存在: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "samples" in data:
        return data["samples"]
    else:
        print("❌ 测试集格式无效 (需要 list 或 {samples: [...]})")
        return []


def get_builtin_samples() -> list:
    """内置评估样例"""
    return [
        {
            "question": "什么是矩阵的秩？如何计算？",
            "answer": "",
            "contexts": [
                "矩阵的秩定义为矩阵中不等于0的子式的最高阶数。",
                "可以通过初等行变换将矩阵化为行阶梯形，非零行数即为秩。",
            ],
            "ground_truth": "矩阵的秩是非零子式的最高阶数，通过初等变换计算非零行数。",
            "metadata": {"subject": "线性代数", "difficulty": "medium"},
        },
        {
            "question": "请解释导数的定义",
            "answer": "",
            "contexts": [
                "导数定义为函数增量与自变量增量之比的极限。",
                "f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
            ],
            "ground_truth": "导数是函数在某点的瞬时变化率，定义为差商的极限。",
            "metadata": {"subject": "高等数学", "difficulty": "easy"},
        },
        {
            "question": "简述 TCP 三次握手的过程",
            "answer": "",
            "contexts": [
                "TCP 三次握手：客户端发送 SYN，服务端回复 SYN-ACK，客户端再发送 ACK。",
                "SYN=1, seq=x → SYN=1, ACK=1, seq=y, ack=x+1 → ACK=1, seq=x+1, ack=y+1",
            ],
            "ground_truth": "客户端→服务端 SYN，服务端→客户端 SYN+ACK，客户端→服务端 ACK。",
            "metadata": {"subject": "计算机网络", "difficulty": "easy"},
        },
    ]


def run_qa_for_samples(samples: list) -> list:
    """
    对每个样本执行完整的问答流程
    生成真实的 answer + contexts 供评估
    """
    enriched = []
    for i, s in enumerate(samples):
        print(f"  处理 {i+1}/{len(samples)}: {s['question'][:50]}...")
        try:
            result = orchestrator.run_sync(
                query=s["question"],
                subject=s.get("metadata", {}).get("subject", ""),
            )
            s["answer"] = result.get("answer", "")
            s["contexts"] = [
                doc.get("text", "")
                for doc in result.get("retrieved_docs", [])
            ]
        except Exception as e:
            print(f"    ⚠️ 问答失败: {e}")
            s["answer"] = f"[错误] {e}"
            s["contexts"] = s.get("contexts", [])
        enriched.append(s)
    return enriched


if __name__ == "__main__":
    main()
