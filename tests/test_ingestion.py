"""
数据摄入模块测试
"""

import pytest

from src.ingestion.chunker import SemanticChunker, TextChunk
from src.ingestion.layout_analyzer import LayoutAnalyzer, BlockType


class TestSemanticChunker:
    """文本分块测试"""

    def test_empty_text(self):
        chunker = SemanticChunker(chunk_size=512)
        chunks = chunker.split("")
        assert len(chunks) == 0

    def test_short_text_single_chunk(self):
        chunker = SemanticChunker(chunk_size=512)
        text = "这是一段简短的文本。"
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        # 生成超过 chunk_size 的文本
        paragraphs = ["段落" + str(i) + "。" * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.split(text)
        # 至少产生多个块
        assert len(chunks) >= 2

    def test_chunk_metadata_preserved(self):
        chunker = SemanticChunker(chunk_size=512)
        text = "这是一段测试文本，包含元数据。"
        metadata = {"subject": "高等数学", "chapter": "第一章"}
        chunks = chunker.split(text, metadata)
        assert len(chunks) > 0
        assert chunks[0].metadata["subject"] == "高等数学"
        assert chunks[0].metadata["chapter"] == "第一章"

    def test_overlap_between_chunks(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=30)
        text = "A. " * 50 + "\n\n" + "B. " * 50 + "\n\n" + "C. " * 50
        chunks = chunker.split(text)

        if len(chunks) >= 2:
            # 验证相邻 chunk 的文本不同 (如果有重叠，最后一个 chunk 的末尾可能出现在下一个的开头)
            assert chunks[0].text != chunks[1].text

    def test_chunk_index_ordering(self):
        chunker = SemanticChunker(chunk_size=100)
        text = "段落1。" * 20 + "\n\n" + "段落2。" * 20 + "\n\n" + "段落3。" * 20
        chunks = chunker.split(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestLayoutAnalyzer:
    """版面分析测试"""

    def test_empty_text(self):
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze("")
        assert result.total_blocks == 0

    def test_title_detection(self):
        analyzer = LayoutAnalyzer()
        text = "第一章 极限与连续\n\n这是正文内容。"
        result = analyzer.analyze(text)
        # 第一章 极限与连续 应被识别为标题
        titles = [b for b in result.blocks if b.block_type == BlockType.TITLE]
        assert len(titles) >= 1

    def test_list_detection(self):
        analyzer = LayoutAnalyzer()
        text = "- 第一项\n- 第二项\n- 第三项"
        result = analyzer.analyze(text)
        list_items = [b for b in result.blocks if b.block_type == BlockType.LIST_ITEM]
        assert len(list_items) == 3

    def test_section_extraction(self):
        analyzer = LayoutAnalyzer()
        text = "第一章 概述\n第一节 背景\n正文内容..."
        result = analyzer.analyze(text)
        assert len(result.sections) >= 2

    def test_formula_detection(self):
        analyzer = LayoutAnalyzer()
        text = "$$f(x) = \\int_a^b g(x)dx$$"
        result = analyzer.analyze(text)
        formulas = [b for b in result.blocks if b.block_type == BlockType.FORMULA]
        assert len(formulas) >= 1

    def test_markdown_output(self):
        analyzer = LayoutAnalyzer()
        text = "第一章 概述\n这是正文。"
        result = analyzer.analyze(text)
        markdown = result.to_markdown()
        assert markdown  # 非空
        assert isinstance(markdown, str)

    def test_document_structure(self):
        analyzer = LayoutAnalyzer()
        text = "第一章 极限\n极限的定义...\n第二章 导数\n导数的定义..."
        sections = analyzer.analyze_document_structure(text)
        assert len(sections) >= 2
