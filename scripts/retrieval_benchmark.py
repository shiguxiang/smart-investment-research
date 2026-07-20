#!/usr/bin/env python3
"""
检索评测 — 大规模评测集 (500+ QA)
基于35份年报自动生成, 计算 BM25 + BGE-Reranker Recall@K

生成策略:
1. 精确指标查询 (公司+指标): ~70条
2. 行业对比查询: ~30条
3. 跨公司对比查询 (同行业内两两组合): ~50条
4. 关键词段落查询 (基于实际提取的文本片段): ~200条
5. 模糊语义查询 (改写+同义词): ~150条
总计: ~500条
"""

import os, sys, json, time, re, random
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from sentence_transformers import CrossEncoder
from src.ingestion.chunker import SemanticChunker
from src.retrieval.keyword_search import BM25Index

PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pdfs')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'retrieval_report.json')


# ============================================================
# Step 1: 解析 + 索引
# ============================================================

def build_index():
    print("[1/5] 解析年报 + 建索引...")
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=64)
    idx = BM25Index()
    pdf_meta = {}

    pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])
    for f in pdfs:
        path = os.path.join(PDF_DIR, f)
        doc = fitz.open(path)
        full_text = "".join(doc[p].get_text("text") + "\n" for p in range(len(doc)))
        doc.close()

        company = f.split('：')[0].strip() if '：' in f else f.replace('.pdf','')
        industry = _classify(f, full_text)
        metrics = _extract_metrics(full_text)
        # 提取实际出现的文本片段 (用于生成段落查询)
        text_samples = _extract_text_samples(full_text)

        chunks = chunker.split(full_text, {
            "file_name": f, "company": company, "industry": industry,
        })
        idx.add_documents([{"id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks])
        pdf_meta[f] = {
            "company": company, "industry": industry,
            "chunks": len(chunks), "metrics": metrics,
            "text_samples": text_samples,
        }

    print(f"  索引: {len(pdfs)} 年报, {idx.doc_count} 文本块")
    return idx, pdf_meta


def _classify(fname, text):
    for ind, kws in [
        ("银行",["银行","工商","招商"]), ("保险",["人寿","保险"]),
        ("石油石化",["石化","石油"]), ("白酒",["茅台","五粮液","汾酒","老窖"]),
        ("新能源",["宁德","比亚迪","隆基"]), ("地产",["万科","保利"]),
        ("家电",["美的","格力"]), ("医药",["恒瑞","药明","迈瑞"]),
        ("半导体",["中芯","京东方"]), ("消费",["中免","牧原"]),
        ("物流",["顺丰"]), ("科技",["海康","金山","汇川"]),
        ("电力",["长江电力"]), ("制造",["绿的","中简"]), ("建材",["海螺"]),
    ]:
        for kw in kws:
            if kw in fname or kw in text[:3000]: return ind
    return "其他"


def _extract_metrics(text):
    result = {}
    for name, pats in {
        "营收": [r'营业收入[约为]?\s*([\d,.]+)\s*[万亿千百]*元',
                  r'营业总收入[约为]?\s*([\d,.]+)'],
        "净利": [r'(?:归属于.{0,10})?净利润[约为]?\s*([\d,.]+)\s*[万亿千百]*元'],
        "ROE": [r'(?:加权平均)?净资产收益率[约为]?\s*([\d.]+)\s*%'],
        "总资产": [r'总资产[约为]?\s*([\d,.]+)\s*[万亿千百]*元'],
    }.items():
        for p in pats:
            m = re.search(p, text)
            if m: result[name] = m.group(1).strip(); break
    return result


def _extract_text_samples(text, max_samples=8):
    """从年报中提取有意义的文本片段作为段落查询依据"""
    samples = []
    # 找含有关键财务术语的段落
    keywords = ["营业收入", "净利润", "毛利率", "资产负债", "现金流", "每股收益",
                "主营业务", "同比增长", "研发投入", "市场占有率", "产能", "分红"]
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50: continue
        for kw in keywords:
            if kw in para:
                samples.append(para[:200])
                break
        if len(samples) >= max_samples: break
    return samples


# ============================================================
# Step 2: 大规模生成 QA 评测集
# ============================================================

def generate_large_test_set(pdf_meta: dict, target: int = 500) -> list:
    print(f"[2/5] 生成大规模QA评测集 (目标 {target}+)...")
    test = []
    meta_list = list(pdf_meta.items())
    random.seed(42)

    # --- 类型1: 精确指标查询 (每个公司×每个指标) ---
    for fname, meta in meta_list:
        c = meta["company"]; ind = meta["industry"]
        for metric_key, metric_label in [("营收","营业收入"), ("净利","净利润"), ("ROE","净资产收益率")]:
            if metric_key in meta["metrics"]:
                test.append({
                    "q": f"{c}2025年{metric_label}是多少？",
                    "gt": fname, "type": "exact_metric"
                })
                # 变体
                variants = [
                    f"{c}去年{metric_label}有多少？",
                    f"查询{c}的{metric_label}数据",
                    f"{c}年报中披露的{metric_label}金额",
                ]
                for v in variants[:2]:
                    test.append({"q": v, "gt": fname, "type": "exact_metric_variant"})

    # --- 类型2: 行业查询 ---
    industries = {}
    for fname, meta in meta_list:
        ind = meta["industry"]
        industries.setdefault(ind, []).append(meta)

    for ind, companies in industries.items():
        if len(companies) >= 2:
            # 行业汇总
            test.append({
                "q": f"{ind}行业2025年整体营收表现如何？",
                "gt": None, "type": "industry_overview",
                "gt_set": [f for f, m in meta_list if m["industry"] == ind]
            })
            test.append({
                "q": f"哪些{ind}公司2025年净利润增长较快？",
                "gt": None, "type": "industry_compare",
                "gt_set": [f for f, m in meta_list if m["industry"] == ind]
            })

    # --- 类型3: 跨公司两两对比 ---
    for ind, companies in industries.items():
        for i in range(len(companies)):
            for j in range(i+1, min(i+3, len(companies))):
                c1 = companies[i]["company"]; c2 = companies[j]["company"]
                f1 = [f for f,m in meta_list if m["company"]==c1][0]
                f2 = [f for f,m in meta_list if m["company"]==c2][0]
                test.append({
                    "q": f"对比{c1}和{c2}2025年盈利能力",
                    "gt": None, "type": "cross_company",
                    "gt_set": [f1, f2]
                })
                test.append({
                    "q": f"{c1}与{c2}谁的营收规模更大？",
                    "gt": None, "type": "cross_company",
                    "gt_set": [f1, f2]
                })

    # --- 类型4: 段落内容查询 (基于实际文本) ---
    for fname, meta in meta_list:
        for sample in meta.get("text_samples", [])[:3]:
            if len(sample) < 60: continue
            # 提取关键短语作为查询
            phrases = re.findall(r'[一-龥]{6,20}', sample)
            for phrase in phrases[:3]:
                if len(phrase) >= 8:
                    test.append({
                        "q": f"关于{phrase}的具体情况",
                        "gt": fname, "type": "paragraph_search"
                    })
                    test.append({
                        "q": f"请说明{phrase}的相关信息",
                        "gt": fname, "type": "paragraph_search"
                    })

    # --- 类型5: 同义改写查询 ---
    rewrite_map = {
        "营收": ["收入", "营业额", "主营业务规模", "销售收入"],
        "净利": ["利润", "盈利", "赚钱", "净利润表现"],
        "增速": ["增长", "涨幅", "提升", "变化趋势"],
    }
    for fname, meta in meta_list:
        c = meta["company"]
        for orig_term, synonyms in rewrite_map.items():
            for syn in synonyms[:2]:
                test.append({
                    "q": f"{c}2025年{syn}是多少？",
                    "gt": fname, "type": "synonym_rewrite"
                })
                test.append({
                    "q": f"{c}的{syn}怎么样？",
                    "gt": fname, "type": "synonym_rewrite"
                })

    # 去重
    seen = set(); unique = []
    for qa in test:
        if qa["q"] not in seen:
            seen.add(qa["q"]); unique.append(qa)

    # 随机采样到目标数量
    if len(unique) > target:
        unique = random.sample(unique, target)

    print(f"  生成 {len(unique)} 条QA (去重后, 含 {len(set(q['type'] for q in unique))} 种类型)")

    # 统计类型分布
    type_counts = {}
    for qa in unique:
        t = qa["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x:-x[1]):
        print(f"    {t}: {c}")

    return unique


# ============================================================
# Step 3: 检索评测
# ============================================================

def evaluate(test_set: list, bm25_idx, reranker_model: str) -> dict:
    print(f"[3/5] 加载 BGE-Reranker: {reranker_model}")
    reranker = CrossEncoder(reranker_model)

    print(f"[4/5] 执行检索评测 ({len(test_set)} 条)...")
    results = []
    bm25_recall = {1:0, 3:0, 5:0, 10:0}
    rerank_recall = {1:0, 3:0, 5:0, 10:0}

    for i, qa in enumerate(test_set):
        query = qa["q"]
        gt_file = qa.get("gt")
        gt_set = qa.get("gt_set", [gt_file] if gt_file else [])

        # BM25 top-50
        kw_results = bm25_idx.search(query, top_k=50)

        # BM25 recall
        kw_ranks = []
        for j, r in enumerate(kw_results):
            fname = r.metadata.get("file_name", "")
            if fname in gt_set:
                kw_ranks.append(j + 1)

        kw_best = min(kw_ranks) if kw_ranks else None
        for k in [1,3,5,10]:
            if kw_best and kw_best <= k: bm25_recall[k] += 1

        # Reranker top-10
        rr_best = None
        if kw_results:
            pairs = [[query, r.text[:512]] for r in kw_results[:30]]
            scores = reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            for rank, idx in enumerate(ranked):
                fname = kw_results[idx].metadata.get("file_name", "")
                if fname in gt_set:
                    rr_best = rank + 1; break

        for k in [1,3,5,10]:
            if rr_best and rr_best <= k: rerank_recall[k] += 1

        results.append({"q": query, "gt_files": gt_set,
                         "bm25_rank": kw_best, "rerank_rank": rr_best})

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(test_set)}  BM25-R@5={bm25_recall[5]/(i+1):.2%}  "
                  f"Rerank-R@5={rerank_recall[5]/(i+1):.2%}")

    total = len(test_set)
    for k in [1,3,5,10]:
        bm25_recall[k] = round(bm25_recall[k] / total, 4)
        rerank_recall[k] = round(rerank_recall[k] / total, 4)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": total,
        "index_size": bm25_idx.doc_count,
        "reranker_model": reranker_model,
        "bm25_only": {f"recall@{k}": bm25_recall[k] for k in [1,3,5,10]},
        "bm25_plus_reranker": {f"recall@{k}": rerank_recall[k] for k in [1,3,5,10]},
        "results": results,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(" 检索评测 — 大规模评测集 + BGE-Reranker")
    print("=" * 60)

    idx, pdf_meta = build_index()
    test_set = generate_large_test_set(pdf_meta, target=500)
    report = evaluate(test_set, idx, "BAAI/bge-reranker-v2-m3")

    # 输出
    print(f"\n[5/5] 评测结果")
    print(f"{'='*60}")
    print(f" 评测集: {report['total_queries']} 条 | 索引: {report['index_size']} 块")
    print(f" 模型: {report['reranker_model']}")
    print()
    print(f" {'指标':<14} {'BM25 Only':>11} {'BM25+Reranker':>13} {'提升':>8}")
    print(f" {'-'*46}")
    for k in [1,3,5,10]:
        b = report["bm25_only"][f"recall@{k}"]
        r = report["bm25_plus_reranker"][f"recall@{k}"]
        lift = f"+{(r-b)*100:.1f}pp" if r > b else f"{(r-b)*100:.1f}pp"
        print(f" {'Recall@'+str(k):<14} {b:>10.2%} {r:>12.2%} {lift:>10}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n 报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
