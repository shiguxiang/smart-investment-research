"""
文档解析 Agent
负责接收用户上传的年报 PDF，调用摄入流水线进行解析
将年报 PDF 转为结构化文本（版面分析 + 表格提取 + 跨页拼接）
"""

import os
from typing import List, Dict, Optional, Any

from src.agents.state import AgentState
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.ppt_parser import PPTParser
from src.ingestion.ocr_processor import OCRProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentAgent:
    """
    文档解析 Agent
    职责: 接收年报 PDF → 版面分析 + 表格提取 + 跨页拼接 → 输出结构化文本
    与财务分析 Agent 解耦，独立处理多模态年报数据
    核心能力: 跨页表格拼接、合并报表识别、财务指标定位
    """

    def __init__(self):
        self.pdf_parser = PDFParser(extract_images=True)
        self.ppt_parser = PPTParser(extract_images=True)
        self.ocr_processor = OCRProcessor(use_gpu=True)

    def process(self, state: AgentState) -> AgentState:
        """
        处理年报解析任务

        输入: state.files (年报 PDF 路径列表)
        输出: state.ocr_text, state.ocr_success, state.ocr_stats
        """
        logger.info(f"文档解析 Agent 开始处理 {len(state.get('files', []))} 份年报")

        state["current_step"] = "document_parsing"
        state["ocr_success"] = False
        all_texts = []
        stats = {
            "total_files": len(state.get("files", [])),
            "success_count": 0,
            "fail_count": 0,
            "total_chars": 0,
            "tables_found": 0,
            "cross_page_merges": 0,
            "file_details": [],
        }

        for file_path in state.get("files", []):
            try:
                file_text, file_stats = self._parse_annual_report(file_path)
                if file_text:
                    all_texts.append(file_text)
                    stats["success_count"] += 1
                    stats["total_chars"] += file_stats.get("chars", 0)
                    stats["tables_found"] += file_stats.get("tables", 0)
                    stats["cross_page_merges"] += file_stats.get("cross_page_merges", 0)
                    stats["file_details"].append(file_stats)
                else:
                    stats["fail_count"] += 1
            except Exception as e:
                logger.error(f"年报解析失败: {file_path}, 错误: {e}")
                stats["fail_count"] += 1
                stats["file_details"].append({
                    "file": os.path.basename(file_path),
                    "success": False,
                    "error": str(e),
                })

        if all_texts:
            state["ocr_text"] = "\n\n---\n\n".join(all_texts)
            state["ocr_success"] = True
            state["ocr_stats"] = stats
            logger.info(
                f"文档解析 Agent 完成: {stats['success_count']}/{stats['total_files']} 成功, "
                f"提取表格 {stats['tables_found']} 张, 跨页拼接 {stats['cross_page_merges']} 处, "
                f"总字符数 {stats['total_chars']}"
            )
        else:
            state["ocr_text"] = ""
            state["ocr_success"] = False
            state["ocr_error"] = "所有年报解析失败"
            state["ocr_stats"] = stats
            logger.warning("文档解析 Agent: 所有年报解析失败")

        return state

    def _parse_annual_report(self, file_path: str) -> tuple:
        """
        解析单份年报
        流程: PDF解析 → 版面分析 → 表格提取 → 跨页拼接 → 结构化输出
        Returns: (text, stats_dict)
        """
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        stats = {
            "file": file_name,
            "type": ext,
            "success": False,
            "chars": 0,
            "tables": 0,
            "cross_page_merges": 0,
        }

        if ext == ".pdf":
            pdf_doc = self.pdf_parser.parse(file_path)
            text_parts = []
            tables_found = 0
            cross_page_merges = 0

            # 收集所有页面的文本和表格
            page_texts = []
            page_tables = []

            for page in pdf_doc.pages:
                # 版面文本
                if page.text:
                    page_texts.append((page.page_num, page.text))

                # 提取表格
                for table_text in page.tables:
                    tables_found += 1
                    page_tables.append((page.page_num, table_text))

                # OCR 处理嵌入图片 (如盖章、签名页)
                for img_bytes in page.images:
                    ocr_result = self.ocr_processor.process_image(
                        img_bytes, page_num=page.page_num
                    )
                    if ocr_result.success and ocr_result.full_text:
                        text_parts.append(
                            f"--- 第 {page.page_num} 页 (扫描件文字) ---\n{ocr_result.full_text}"
                        )

            # 跨页表格拼接 (检测连续页面的表格并合并)
            merged_tables = self._merge_cross_page_tables(page_tables)
            cross_page_merges = len(page_tables) - len(merged_tables)

            # 组装输出
            for page_num, text in page_texts:
                text_parts.append(f"--- 第 {page_num} 页 ---\n{text}")

            for table_info in merged_tables:
                if isinstance(table_info, tuple):
                    # 多页合并的表格
                    pages, table_text = table_info
                    text_parts.append(
                        f"--- 第 {pages[0]}-{pages[-1]} 页 (合并表格) ---\n{table_text}"
                    )
                else:
                    page_num, table_text = table_info
                    text_parts.append(
                        f"--- 第 {page_num} 页 (财务表格) ---\n{table_text}"
                    )

            full_text = "\n\n".join(text_parts)
            stats["chars"] = len(full_text)
            stats["pages"] = pdf_doc.total_pages
            stats["tables"] = tables_found
            stats["cross_page_merges"] = cross_page_merges
            stats["success"] = True

            return full_text, stats

        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
            # 扫描版年报图片 OCR
            ocr_result = self.ocr_processor.process_image_file(file_path)
            if ocr_result.success:
                stats["chars"] = len(ocr_result.full_text)
                stats["ocr_blocks"] = len(ocr_result.results)
                stats["success"] = True
                return ocr_result.full_text, stats
            else:
                stats["error"] = ocr_result.error
                return "", stats

        else:
            stats["error"] = f"不支持的文件类型: {ext}"
            logger.warning(f"不支持的文件类型: {ext}")
            return "", stats

    def _merge_cross_page_tables(
        self, page_tables: List[tuple]
    ) -> List:
        """
        跨页表格拼接
        检测连续页面的表格是否为同一张表（通过表头匹配判断），自动合并
        """
        if len(page_tables) <= 1:
            return page_tables

        merged = []
        current_group = [page_tables[0]]

        for i in range(1, len(page_tables)):
            prev_page, prev_table = page_tables[i - 1]
            curr_page, curr_table = page_tables[i]

            # 判断是否为跨页的同一张表:
            # 1. 页面连续
            # 2. 当前页表格没有表头（或表头与前页一致）
            is_continuation = (
                curr_page == prev_page + 1
                and not self._has_table_header(curr_table)
            )

            if is_continuation:
                current_group.append(page_tables[i])
            else:
                # 保存当前组
                merged.append(self._flatten_group(current_group))
                current_group = [page_tables[i]]

        merged.append(self._flatten_group(current_group))
        return merged

    @staticmethod
    def _has_table_header(table_text: str) -> bool:
        """判断表格文本是否包含表头（如: 项目、金额、本期、上期等）"""
        header_keywords = ["项目", "金额", "本期", "上期", "合计", "附注", "科目", "余额"]
        first_line = table_text.split("\n")[0] if table_text else ""
        return any(kw in first_line for kw in header_keywords)

    @staticmethod
    def _flatten_group(group: List[tuple]) -> tuple:
        """将同组（跨页）表格合并"""
        if len(group) == 1:
            return group[0]

        pages = tuple(item[0] for item in group)
        merged_text = "\n(续)\n".join(item[1] for item in group)
        return (pages, merged_text)

    def process_with_ingestion_pipeline(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        使用完整摄入流水线处理年报 (含分块)
        用于数据导入场景
        """
        pipeline = IngestionPipeline()
        result = pipeline.ingest_file(file_path, metadata or {})

        return {
            "success": result.success,
            "file_name": result.file_name,
            "file_type": result.file_type,
            "chunks_count": len(result.chunks),
            "stats": result.stats,
            "error": result.error,
        }
