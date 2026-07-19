#!/usr/bin/env python3
"""
批量数据导入脚本
将 data/ 目录下的年报PDF/PPT/扫描件批量解析并写入 Milvus

用法:
    python scripts/ingest.py --dir data/pdfs --subject "高等数学" --chapter "第一章"
    python scripts/ingest.py --file "data/exams/midterm.pdf" --subject "线性代数"
"""

import os
import sys
import json
import time
import argparse

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pipeline import IngestionPipeline, IngestResult
from src.retrieval.milvus_client import milvus_client
from src.retrieval.embedder import Embedder
from src.retrieval.keyword_search import bm25_index
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="智能投研分析系统 — 年报数据导入工具")

    parser.add_argument(
        "--dir",
        type=str,
        help="要导入的目录路径",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="要导入的单个文件路径",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="科目名称 (如 '高等数学')",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default="",
        help="章节 (如 '第一章 极限与连续')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="分块大小 (默认512字符)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="分块重叠 (默认64字符)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="禁用 OCR (仅提取原生文本)",
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default="",
        help="保存 BM25 索引到指定路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出结果 JSON 文件路径",
    )

    args = parser.parse_args()

    # 验证参数
    if not args.dir and not args.file:
        parser.error("请指定 --dir 或 --file")
    if args.dir and args.file:
        parser.error("--dir 和 --file 不能同时使用")

    # 初始化
    logger.info("=" * 50)
    logger.info("数据导入开始")
    logger.info(f"科目: {args.subject}")
    if args.chapter:
        logger.info(f"章节: {args.chapter}")
    logger.info("=" * 50)

    # 连接 Milvus
    print("\n[1/4] 连接 Milvus...")
    if not milvus_client.connect():
        print("❌ Milvus 连接失败，退出")
        sys.exit(1)

    collection = milvus_client.setup()
    print(f"✓ Milvus 就绪 — Collection: {collection.name}")

    # 初始化组件
    print("\n[2/4] 初始化解析组件...")
    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_use_gpu=True,
        extract_images=not args.no_ocr,
    )
    embedder = Embedder()
    print("✓ 组件就绪")

    # 执行摄入
    print("\n[3/4] 处理文件...")
    metadata = {
        "subject": args.subject,
        "chapter": args.chapter,
    }

    all_chunks = []
    if args.file:
        result = pipeline.ingest_file(args.file, metadata)
        results = [result]
        all_chunks.extend(result.chunks)
        print(f"  文件: {result.file_name} — {'✓' if result.success else '✗'}")
        if result.stats:
            print(f"  统计: {json.dumps(result.stats, ensure_ascii=False)}")
    else:
        results = pipeline.ingest_directory(args.dir, metadata)
        for r in results:
            all_chunks.extend(r.chunks)

    success_count = sum(1 for r in results if r.success)
    print(f"\n✓ 处理完成: {success_count}/{len(results)} 成功")
    print(f"  总文本块: {len(all_chunks)}")

    if not all_chunks:
        print("❌ 没有生成任何文本块，退出")
        sys.exit(1)

    # 向量化 + 写入 Milvus
    print(f"\n[4/4] 向量化并写入 Milvus ({len(all_chunks)} 条)...")

    texts = [chunk.text for chunk in all_chunks]
    print(f"  正在向量化... (这可能需要几分钟)")

    start = time.time()
    embeddings = embedder.embed_batch(texts)
    embed_time = time.time() - start

    success_embeds = sum(1 for e in embeddings if e is not None)
    print(f"  向量化耗时: {embed_time:.1f}s, 成功: {success_embeds}/{len(texts)}")

    # 构建 Milvus 数据
    milvus_data = []
    for i, chunk in enumerate(all_chunks):
        embedding = embeddings[i] if i < len(embeddings) else None
        if embedding is None:
            print(f"  跳过 chunk_{i} (向量化失败)")
            continue

        milvus_data.append({
            "id": chunk.chunk_id,
            "text": chunk.text,
            "embedding": embedding,
            "file_name": chunk.metadata.get("file_name", ""),
            "file_type": chunk.metadata.get("file_type", ""),
            "subject": chunk.metadata.get("subject", args.subject),
            "chapter": chunk.metadata.get("chapter", args.chapter),
            "chunk_index": chunk.chunk_index,
            "char_count": len(chunk.text),
            "source": chunk.metadata.get("source_dir", ""),
        })

    if milvus_data:
        inserted = milvus_client.insert(milvus_data)
        print(f"✓ Milvus 写入完成: {inserted} 条")
    else:
        print("❌ 没有有效数据可写入")
        sys.exit(1)

    # 构建 BM25 索引
    print(f"\n构建 BM25 关键词索引...")
    bm25_index.build_from_chunks(all_chunks)
    print(f"✓ BM25 索引: {bm25_index.doc_count} 篇文档")

    # 可选: 保存索引
    if args.save_index:
        bm25_index.save(args.save_index)
        print(f"✓ BM25 索引已保存: {args.save_index}")

    # 可选: 输出结果
    if args.output:
        output_data = {
            "total_files": len(results),
            "success_count": success_count,
            "total_chunks": len(all_chunks),
            "milvus_inserted": len(milvus_data),
            "subject": args.subject,
            "chapter": args.chapter,
            "results": [
                {
                    "file_name": r.file_name,
                    "file_type": r.file_type,
                    "success": r.success,
                    "chunks": len(r.chunks),
                    "stats": r.stats,
                    "error": r.error,
                }
                for r in results
            ],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {args.output}")

    # 完成
    print("\n" + "=" * 50)
    print(f"✓ 导入完成!")
    print(f"  科目: {args.subject}")
    print(f"  文件: {success_count}/{len(results)}")
    print(f"  文本块: {len(all_chunks)}")
    print(f"  Milvus: {len(milvus_data)} 条向量")
    print(f"  BM25: {bm25_index.doc_count} 篇文档")
    print("=" * 50)


if __name__ == "__main__":
    main()
