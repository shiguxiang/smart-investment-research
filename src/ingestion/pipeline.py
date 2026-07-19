"""
数据摄入流水线
编排 PDF/PPT/Image 的完整处理流程: 解析 → OCR → 版面分析 → 分块 → 入库
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from src.ingestion.pdf_parser import PDFParser, PDFDocument
from src.ingestion.ppt_parser import PPTParser, PPTDocument
from src.ingestion.ocr_processor import OCRProcessor, OcrPageResult
from src.ingestion.layout_analyzer import LayoutAnalyzer, LayoutResult
from src.ingestion.chunker import SemanticChunker, TextChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """摄入结果"""
    file_path: str
    file_name: str
    file_type: str  # pdf / ppt / image
    success: bool
    chunks: List[TextChunk] = field(default_factory=list)
    markdown_text: str = ""
    error: Optional[str] = None
    stats: Dict = field(default_factory=dict)


class IngestionPipeline:
    """
    数据摄入流水线
    统一处理年报 PDF / PPT / 扫描件 三类投研资料
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".ppt": "ppt",
        ".pptx": "ppt",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".bmp": "image",
        ".tiff": "image",
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        ocr_use_gpu: bool = True,
        extract_images: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化各子组件
        self.pdf_parser = PDFParser(extract_images=extract_images)
        self.ppt_parser = PPTParser(extract_images=extract_images)
        self.ocr_processor = OCRProcessor(use_gpu=ocr_use_gpu)
        self.layout_analyzer = LayoutAnalyzer()
        self.chunker = SemanticChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # 进度回调
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """设置进度回调"""
        self._progress_callback = callback

    def _report_progress(self, message: str, progress: float):
        """报告进度"""
        if self._progress_callback:
            self._progress_callback(message, progress)
        logger.info(f"[{progress:.0%}] {message}")

    def detect_file_type(self, file_path: str) -> Optional[str]:
        """检测文件类型"""
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext)

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
    ) -> IngestResult:
        """
        摄入单个文件

        Args:
            file_path: 文件路径
            metadata: 附加元数据 (科目、章节、学期等)

        Returns:
            IngestResult 包含所有分块结果
        """
        file_type = self.detect_file_type(file_path)

        if file_type is None:
            return IngestResult(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_type="unknown",
                success=False,
                error=f"不支持的文件类型: {os.path.splitext(file_path)[1]}",
            )

        self._report_progress(f"开始处理: {os.path.basename(file_path)}", 0.0)

        try:
            if file_type == "pdf":
                return self._ingest_pdf(file_path, metadata or {})
            elif file_type == "ppt":
                return self._ingest_ppt(file_path, metadata or {})
            elif file_type == "image":
                return self._ingest_image(file_path, metadata or {})
        except Exception as e:
            logger.error(f"文件摄入失败: {file_path}, 错误: {e}")
            return IngestResult(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_type=file_type,
                success=False,
                error=str(e),
            )

    def ingest_directory(
        self,
        dir_path: str,
        metadata: Optional[Dict] = None,
        recursive: bool = True,
    ) -> List[IngestResult]:
        """
        批量摄入目录下的所有支持文件

        Args:
            dir_path: 目录路径
            metadata: 全局元数据
            recursive: 是否递归子目录

        Returns:
            IngestResult 列表
        """
        results = []
        pattern = "**/*" if recursive else "*"

        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(Path(dir_path).glob(f"{pattern}{ext}"))

        total = len(files)
        self._report_progress(f"发现 {total} 个待处理文件", 0.0)

        for i, file_path in enumerate(files):
            file_meta = metadata.copy() if metadata else {}
            file_meta["source_dir"] = str(file_path.parent)

            # 从路径推断科目/章节
            relative = file_path.relative_to(dir_path)
            parts = relative.parts
            if len(parts) > 1:
                file_meta["subject"] = parts[0]  # 第一级目录作为科目
            if len(parts) > 2:
                file_meta["chapter"] = parts[1]  # 第二级目录作为章节

            result = self.ingest_file(str(file_path), file_meta)
            results.append(result)

            progress = (i + 1) / total
            self._report_progress(
                f"已处理 {i + 1}/{total}: {result.file_name} "
                f"({'✓' if result.success else '✗'})",
                progress,
            )

        success_count = sum(1 for r in results if r.success)
        logger.info(f"批量摄入完成: {success_count}/{total} 成功")
        return results

    def _ingest_pdf(
        self, file_path: str, metadata: Dict
    ) -> IngestResult:
        """处理 PDF 文件"""
        # 1. 解析 PDF
        pdf_doc = self.pdf_parser.parse(file_path)

        # 2. OCR 处理嵌入图片
        all_text_parts = []
        ocr_results = []

        for page in pdf_doc.pages:
            # 页面文本
            if page.text:
                all_text_parts.append(f"[第 {page.page_num} 页]\n{page.text}")

            # 嵌入图片 OCR
            for img_bytes in page.images:
                ocr_result = self.ocr_processor.process_image(
                    img_bytes, page_num=page.page_num
                )
                ocr_results.append(ocr_result)
                if ocr_result.full_text:
                    all_text_parts.append(
                        f"[第 {page.page_num} 页 - 图片文字]\n{ocr_result.full_text}"
                    )

            # 表格文本
            for table_text in page.tables:
                all_text_parts.append(
                    f"[第 {page.page_num} 页 - 表格]\n{table_text}"
                )

        # 3. 合并全文
        full_text = "\n\n".join(all_text_parts)

        # 4. 版面分析
        layout = self.layout_analyzer.analyze(full_text, source=pdf_doc.file_name)

        # 5. 分块
        enhanced_metadata = {
            **metadata,
            "file_type": "pdf",
            "file_name": pdf_doc.file_name,
            "total_pages": pdf_doc.total_pages,
            "sections": layout.sections,
        }
        chunks = self.chunker.split(full_text, enhanced_metadata)

        return IngestResult(
            file_path=file_path,
            file_name=pdf_doc.file_name,
            file_type="pdf",
            success=True,
            chunks=chunks,
            markdown_text=layout.to_markdown(),
            stats={
                "pages": pdf_doc.total_pages,
                "ocr_images": len(ocr_results),
                "ocr_chars": sum(len(r.full_text) for r in ocr_results),
                "total_chars": len(full_text),
                "chunks": len(chunks),
                "sections": len(layout.sections),
            },
        )

    def _ingest_ppt(
        self, file_path: str, metadata: Dict
    ) -> IngestResult:
        """处理 PPT 文件"""
        # 1. 解析 PPT
        ppt_doc = self.ppt_parser.parse(file_path)

        # 2. 转换为 Markdown
        markdown = self.ppt_parser.to_markdown(ppt_doc)

        # 3. OCR 处理嵌入图片
        ocr_texts = []
        ocr_count = 0
        for slide in ppt_doc.slides:
            for img_bytes in slide.images:
                ocr_result = self.ocr_processor.process_image(
                    img_bytes, page_num=slide.slide_num
                )
                ocr_count += 1
                if ocr_result.full_text:
                    ocr_texts.append(
                        f"[幻灯片 {slide.slide_num} - 图片文字]\n{ocr_result.full_text}"
                    )

        # 4. 合并文本
        full_text = markdown
        if ocr_texts:
            full_text += "\n\n" + "\n\n".join(ocr_texts)

        # 5. 版面分析
        layout = self.layout_analyzer.analyze(full_text, source=ppt_doc.file_name)

        # 6. 分块
        enhanced_metadata = {
            **metadata,
            "file_type": "ppt",
            "file_name": ppt_doc.file_name,
            "total_slides": ppt_doc.total_slides,
        }
        chunks = self.chunker.split(full_text, enhanced_metadata)

        return IngestResult(
            file_path=file_path,
            file_name=ppt_doc.file_name,
            file_type="ppt",
            success=True,
            chunks=chunks,
            markdown_text=markdown,
            stats={
                "slides": ppt_doc.total_slides,
                "ocr_images": ocr_count,
                "total_chars": len(full_text),
                "chunks": len(chunks),
                "sections": len(layout.sections),
            },
        )

    def _ingest_image(
        self, file_path: str, metadata: Dict
    ) -> IngestResult:
        """处理单张图片 (年报扫描件等)"""
        file_name = os.path.basename(file_path)

        # 1. OCR 识别
        ocr_result = self.ocr_processor.process_image_file(file_path)

        if not ocr_result.success:
            return IngestResult(
                file_path=file_path,
                file_name=file_name,
                file_type="image",
                success=False,
                error=ocr_result.error or "OCR 识别失败",
            )

        # 2. 版面分析
        layout = self.layout_analyzer.analyze(
            ocr_result.full_text, source=file_name
        )

        # 3. 分块
        enhanced_metadata = {
            **metadata,
            "file_type": "image",
            "file_name": file_name,
            "ocr_blocks": len(ocr_result.results),
            "formulas": ocr_result.formulas,
        }
        chunks = self.chunker.split(ocr_result.full_text, enhanced_metadata)

        return IngestResult(
            file_path=file_path,
            file_name=file_name,
            file_type="image",
            success=True,
            chunks=chunks,
            markdown_text=layout.to_markdown(),
            stats={
                "ocr_blocks": len(ocr_result.results),
                "formulas": len(ocr_result.formulas),
                "total_chars": len(ocr_result.full_text),
                "chunks": len(chunks),
                "avg_confidence": (
                    sum(r.confidence for r in ocr_result.results) / len(ocr_result.results)
                    if ocr_result.results
                    else 0
                ),
            },
        )
