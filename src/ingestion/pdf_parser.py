"""
PDF 解析器
基于 PyMuPDF (fitz) 提取文字、图片区域和元数据
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PDFPageContent:
    """单页 PDF 内容"""
    page_num: int
    text: str = ""
    images: List[bytes] = field(default_factory=list)
    image_bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)


@dataclass
class PDFDocument:
    """PDF 文档解析结果"""
    file_path: str
    file_name: str
    total_pages: int
    pages: List[PDFPageContent] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class PDFParser:
    """PDF 解析器 - 提取文字 + 图片区域"""

    def __init__(self, extract_images: bool = True, dpi: int = 200):
        """
        Args:
            extract_images: 是否提取嵌入图片
            dpi: 图片提取分辨率
        """
        self.extract_images = extract_images
        self.dpi = dpi

    def parse(self, file_path: str) -> PDFDocument:
        """
        解析 PDF 文件

        Args:
            file_path: PDF 文件路径

        Returns:
            PDFDocument 包含所有页面的文字和图片信息
        """
        logger.info(f"开始解析 PDF: {file_path}")

        doc = fitz.open(file_path)

        pdf_doc = PDFDocument(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            total_pages=len(doc),
            metadata=dict(doc.metadata),
        )

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_content = self._parse_page(page, page_idx)
            pdf_doc.pages.append(page_content)

        doc.close()
        logger.info(
            f"PDF 解析完成: {pdf_doc.file_name}, "
            f"共 {pdf_doc.total_pages} 页, "
            f"总字符数 {sum(len(p.text) for p in pdf_doc.pages)}"
        )

        return pdf_doc

    def _parse_page(self, page: fitz.Page, page_idx: int) -> PDFPageContent:
        """解析单页内容"""
        content = PDFPageContent(page_num=page_idx + 1)

        # 1. 提取文本
        text = page.get_text("text")
        content.text = text.strip() if text else ""

        # 2. 提取图片
        if self.extract_images:
            image_list = page.get_images(full=True)
            for img_info in image_list:
                try:
                    xref = img_info[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    content.images.append(image_bytes)

                    # 获取图片在页面中的位置
                    rects = page.get_image_rects(img_info)
                    if rects:
                        bbox = rects[0]
                        content.image_bboxes.append(
                            (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                        )
                except Exception as e:
                    logger.warning(f"提取图片失败 (page {page_idx}, xref {xref}): {e}")

        # 3. 提取表格 (简单检测 - 通过高密度文本行)
        content.tables = self._detect_tables(page)

        return content

    def _detect_tables(self, page: fitz.Page) -> List[str]:
        """检测页面中的表格区域 (启发式)"""
        tables = []
        try:
            # 用 dict 模式提取带位置信息的文本块
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # text block
                    lines = block.get("lines", [])
                    if len(lines) >= 3:  # 3行以上可能是表格
                        # 检查是否有多列对齐
                        if self._is_table_like(lines):
                            text = ""
                            for line in lines:
                                spans = line.get("spans", [])
                                line_text = " | ".join(
                                    s["text"].strip() for s in spans if s["text"].strip()
                                )
                                text += line_text + "\n"
                            tables.append(text.strip())
        except Exception as e:
            logger.debug(f"表格检测异常: {e}")

        return tables

    @staticmethod
    def _is_table_like(lines: List[Dict]) -> bool:
        """简单判断是否为表格结构"""
        if len(lines) < 2:
            return False
        # 检查每行的 span 数量是否一致
        span_counts = [len(line.get("spans", [])) for line in lines]
        avg_spans = sum(span_counts) / len(span_counts)
        return avg_spans >= 2

    def extract_full_text(self, file_path: str) -> str:
        """提取 PDF 全部文本 (用于快速预览)"""
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()
        return full_text.strip()

    @staticmethod
    def get_page_count(file_path: str) -> int:
        """获取 PDF 页数"""
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count
