"""
智能文本分块模块
按语义段落分块，支持滑动窗口重叠，保留文档元数据
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """文本块"""
    chunk_id: str
    text: str
    metadata: Dict = field(default_factory=dict)
    chunk_index: int = 0
    start_pos: int = 0  # 在原文本中的起始位置
    end_pos: int = 0    # 在原文本中的结束位置


class SemanticChunker:
    """
    语义分块器
    按自然段落边界切分，保持语义完整性
    """

    # 中文段落分隔符
    SENTENCE_SEPARATORS = [
        r"\n{2,}",           # 多个换行
        r"\n(?=[一二三四五六七八九十\d]+[、．.])",  # 标题前换行
        r"(?<=[。！？!?])\n", # 句末换行
        r"\n(?=第[一二三四五六七八九十\d]+[章节])",  # "第X章/节" 前
        r"(?<=\n)#{1,3}\s",   # Markdown 标题后
    ]

    # 不可分割的边界 (不在这些位置切分)
    NO_SPLIT_PATTERNS = [
        r"…\n",              # 省略号后
        r"[，,；;：:]\n",     # 逗号/分号/冒号后
        r"[（(]\n",           # 开括号后
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Args:
            chunk_size: 每个分块的目标大小 (字符数)
            chunk_overlap: 相邻分块的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        将文本分割为语义块

        Args:
            text: 输入文本
            metadata: 附加元数据 (来源、科目、章节等)

        Returns:
            TextChunk 列表
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        # Step 1: 按段落分割
        paragraphs = self._split_by_paragraph(text)

        # Step 2: 合并段落为 chunk_size 大小的块 (带重叠)
        chunks = self._merge_paragraphs(paragraphs, metadata)

        logger.info(
            f"文本分块完成: {len(paragraphs)} 段落 → {len(chunks)} 块 "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

        return chunks

    def _split_by_paragraph(self, text: str) -> List[str]:
        """按自然段落边界分割"""
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 先用双换行分隔
        paragraphs = re.split(r"\n{2,}", text)

        # 进一步拆分过长段落
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= self.chunk_size * 1.5:
                result.append(para)
            else:
                # 长段落按句子切分
                sub_paras = self._split_long_paragraph(para)
                result.extend(sub_paras)

        return result

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """拆分过长段落"""
        # 按中文句号、问号、感叹号切分
        sentences = re.split(r"(?<=[。！？!?])", paragraph)

        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [paragraph]

    def _merge_paragraphs(
        self, paragraphs: List[str], metadata: Dict
    ) -> List[TextChunk]:
        """合并段落为固定大小的块 (滑动窗口重叠)"""
        chunks = []

        # 构建带位置信息的段落列表
        pos = 0
        indexed_paras = []
        for para in paragraphs:
            start = pos
            end = pos + len(para)
            indexed_paras.append((para, start, end))
            pos = end + 2  # +2 补偿分隔符

        chunk_index = 0
        current_text = ""
        current_start = 0
        prev_overlap_text = ""

        for para, p_start, p_end in indexed_paras:
            if not current_text:
                # 新块开始: 加上前一块的重叠文本
                if prev_overlap_text:
                    current_text = prev_overlap_text
                    current_start = max(0, p_start - len(prev_overlap_text))
                else:
                    current_text = para
                    current_start = p_start

            elif len(current_text) + len(para) + 1 <= self.chunk_size:
                current_text += "\n" + para
            else:
                # 当前块已满，保存
                chunk = TextChunk(
                    chunk_id=f"chunk_{chunk_index:05d}",
                    text=current_text.strip(),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_count": len(current_text),
                    },
                    chunk_index=chunk_index,
                    start_pos=current_start,
                    end_pos=current_start + len(current_text),
                )
                chunks.append(chunk)

                # 准备下一个块 (带重叠)
                overlap_text = self._get_overlap(current_text, para)
                current_text = overlap_text + para if overlap_text else para
                current_start = max(0, p_start - len(overlap_text))
                prev_overlap_text = overlap_text
                chunk_index += 1

        # 最后一个块
        if current_text.strip():
            chunk = TextChunk(
                chunk_id=f"chunk_{chunk_index:05d}",
                text=current_text.strip(),
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "char_count": len(current_text),
                },
                chunk_index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_text),
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap(self, full_text: str, next_para: str) -> str:
        """获取与下一个段落的语义重叠部分"""
        if self.chunk_overlap <= 0:
            return ""

        # 取全文末尾的 overlap 字符
        if len(full_text) <= self.chunk_overlap:
            return full_text + "\n"

        # 从 overlap 边界开始找最近的句子边界
        overlap_start = len(full_text) - self.chunk_overlap
        overlap_text = full_text[overlap_start:]

        # 尝试从最近句号处开始
        last_period = max(
            overlap_text.rfind("。"),
            overlap_text.rfind("！"),
            overlap_text.rfind("？"),
            overlap_text.rfind("\n"),
        )
        if last_period > 0:
            return overlap_text[last_period + 1:]

        return overlap_text


class RecursiveChunker:
    """
    递归分块器 (fallback)
    当语义分块效果不佳时的备选方案
    """

    SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """递归分割文本"""
        metadata = metadata or {}
        segments = self._recursive_split(text)
        chunks = []

        for i, seg in enumerate(segments):
            chunk = TextChunk(
                chunk_id=f"chunk_{i:05d}",
                text=seg.strip(),
                metadata={**metadata, "chunk_index": i, "char_count": len(seg)},
                chunk_index=i,
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, depth: int = 0) -> List[str]:
        """递归分割"""
        if depth >= len(self.SEPARATORS):
            return [text]

        separator = self.SEPARATORS[depth]
        if not separator:
            return [text]

        if len(text) <= self.chunk_size:
            return [text]

        splits = text.split(separator)
        result = []

        current = ""
        for s in splits:
            if len(current) + len(s) + len(separator) <= self.chunk_size:
                current += (separator if current else "") + s
            else:
                if current:
                    result.append(current)
                # 如果单个片段就超过 chunk_size，递归处理
                if len(s) > self.chunk_size:
                    result.extend(self._recursive_split(s, depth + 1))
                else:
                    current = s

        if current:
            result.append(current)

        return result
