#!/usr/bin/env python3
"""
年报解析评测脚本
遍历 data/pdfs 下所有年报，测试:
1. 解析成功率 + 首字响应时间
2. 文本提取量 + 表格检测数
3. 核心财务指标提取
4. 输出评测报告 JSON
"""

import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from src.ingestion.layout_analyzer import LayoutAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)

PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_report.json')

# 核心财务指标正则
METRIC_PATTERNS = {
    "营业收入": [
        r'营业收入[约为]?\s*[\d,.]+\s*[万亿千百]*元',
        r'营业总收入[约为]?\s*[\d,.]+\s*[万亿千百]*元',
        r'实现营收[约为]?\s*[\d,.]+\s*[万亿千百]*元',
    ],
    "净利润": [
        r'净利润[约为]?\s*[（(]?[-\d,.]+\s*[万亿千百]*元',
        r'归[属于母]*净利润[约为]?\s*[\d,.]+\s*[万亿千百]*元',
    ],
    "ROE": [
        r'(?:加权平均)?净资产收益率[约为]?\s*[\d.]+\s*%',
        r'ROE[约为]?\s*[\d.]+\s*%',
    ],
    "总资产": [
        r'总资产[约为]?\s*[\d,.]+\s*[万亿千百]*元',
        r'资产总[计额][约为]?\s*[\d,.]+\s*[万亿千百]*元',
    ],
    "每股收益": [
        r'每股收益[约为]?\s*[\d.]+\s*元',
        r'基本每股收益[约为]?\s*[\d.]+\s*元',
    ],
    "毛利率": [
        r'毛利率[约为]?\s*[\d.]+\s*%',
        r'综合毛利率[约为]?\s*[\d.]+\s*%',
    ],
}


def parse_one(file_path: str) -> dict:
    """解析单份年报, 返回统计 + 指标"""
    file_name = os.path.basename(file_path)
    size_mb = os.path.getsize(file_path) / 1024 / 1024

    result = {
        "file": file_name,
        "size_mb": round(size_mb, 1),
        "success": False,
        "pages": 0,
        "parse_time_s": 0,
        "chars": 0,
        "tables_found": 0,
        "sections": 0,
        "metrics": {},
        "error": None,
    }

    try:
        t0 = time.time()
        doc = fitz.open(file_path)
        pages = len(doc)
        full_text = ""
        tables_found = 0

        for p in range(pages):
            page = doc[p]
            text = page.get_text("text")
            full_text += text + "\n"

            # 表格检测 (多span对齐)
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b.get("type") == 0:
                    lines = b.get("lines", [])
                    if len(lines) >= 3:
                        counts = [len(l.get("spans", [])) for l in lines]
                        if sum(counts) / len(counts) >= 3:
                            tables_found += 1

        doc.close()
        elapsed = time.time() - t0

        # 版面分析
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(full_text, source=file_name)

        # 财务指标提取
        metrics = {}
        for name, patterns in METRIC_PATTERNS.items():
            for pat in patterns:
                match = re.search(pat, full_text)
                if match:
                    metrics[name] = match.group(0).strip()
                    break

        result.update({
            "success": True,
            "pages": pages,
            "parse_time_s": round(elapsed, 3),
            "chars": len(full_text),
            "tables_found": tables_found,
            "sections": len(layout.sections),
            "metrics": metrics,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


def run_benchmark():
    """全量评测"""
    pdfs = sorted([
        f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')
    ])

    print(f"年报总数: {len(pdfs)}")
    print(f"{'='*60}")

    results = []
    success = 0
    fail = 0
    total_chars = 0
    total_tables = 0
    total_time = 0
    metrics_found = {}
    ttfs = []  # time-to-first-success

    for i, f in enumerate(pdfs):
        path = os.path.join(PDF_DIR, f)
        r = parse_one(path)
        results.append(r)

        status = "OK" if r["success"] else "FAIL"
        if r["success"]:
            success += 1
            total_chars += r["chars"]
            total_tables += r["tables_found"]
            total_time += r["parse_time_s"]
            ttfs.append(r["parse_time_s"])
            for k in r["metrics"]:
                metrics_found[k] = metrics_found.get(k, 0) + 1

        else:
            fail += 1

        print(f"  [{status}] {r['file'][:50]:50s} "
              f"{r['pages']:3d}p  {r['parse_time_s']:.2f}s  "
              f"{r['chars']:6d}chars  tables:{r['tables_found']}  "
              f"metrics:{len(r['metrics'])}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"评测结果汇总")
    print(f"{'='*60}")
    print(f"  成功率: {success}/{len(pdfs)} ({success/len(pdfs)*100:.0f}%)")
    if ttfs:
        ttfs.sort()
        print(f"  首字响应时间: avg={sum(ttfs)/len(ttfs):.2f}s, "
              f"P50={ttfs[len(ttfs)//2]:.2f}s, P95={ttfs[int(len(ttfs)*0.95)]:.2f}s")
    print(f"  总文本量: {total_chars:,} 字符")
    print(f"  总表格数: {total_tables}")
    print(f"  总解析耗时: {total_time:.1f}s")
    print(f"  财务指标提取覆盖率:")
    for k, v in sorted(metrics_found.items(), key=lambda x: -x[1]):
        pct = v / len(pdfs) * 100
        bar = "#" * int(pct / 5)
        print(f"    {k:10s} {v:2d}/{len(pdfs)} ({pct:3.0f}%) {bar}")

    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(pdfs),
        "success": success,
        "fail": fail,
        "success_rate": f"{success/len(pdfs)*100:.0f}%",
        "ttf_avg_s": round(sum(ttfs)/len(ttfs), 2) if ttfs else 0,
        "ttf_p50_s": round(ttfs[len(ttfs)//2], 2) if ttfs else 0,
        "ttf_p95_s": round(ttfs[int(len(ttfs)*0.95)], 2) if ttfs else 0,
        "total_chars": total_chars,
        "total_tables": total_tables,
        "total_parse_time_s": round(total_time, 1),
        "metrics_coverage": metrics_found,
        "results": results,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评测报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    run_benchmark()
