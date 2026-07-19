"""
OCR 处理模块
基于 PaddleOCR 识别图片/PDF 截图中的文字
同时处理公式区域的识别标注
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """OCR 识别结果"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    block_type: str = "text"  # text / formula / table


@dataclass
class OcrPageResult:
    """单页 OCR 结果"""
    page_num: int = 0
    results: List[OCRResult] = field(default_factory=list)
    full_text: str = ""
    formulas: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class OCRProcessor:
    """
    OCR 处理器
    封装 PaddleOCR，支持中英文混合识别和公式区域标注
    """

    # PaddleOCR 中英文模型
    DEFAULT_LANG = "ch"

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "ch",
        enable_formula_detection: bool = True,
    ):
        """
        Args:
            use_gpu: 是否使用 GPU 推理
            lang: 识别语言 ("ch" 中英文混合)
            enable_formula_detection: 是否启用公式区域检测
        """
        self.use_gpu = use_gpu
        self.enable_formula_detection = enable_formula_detection

        self._ocr = None  # 延迟初始化

    @property
    def ocr(self):
        """延迟加载 PaddleOCR (首次调用时初始化)"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.DEFAULT_LANG,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
                logger.info("PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
                raise RuntimeError(f"PaddleOCR 初始化失败: {e}")
        return self._ocr

    def process_image(
        self, image: bytes, page_num: int = 0
    ) -> OcrPageResult:
        """
        对单张图片执行 OCR

        Args:
            image: 图片字节数据
            page_num: 页码 (用于结果标注)

        Returns:
            OcrPageResult 包含识别文本和位置信息
        """
        page_result = OcrPageResult(page_num=page_num)

        try:
            # 将 bytes 转为 PIL Image 再转 numpy array
            pil_image = Image.open(tempfile.TemporaryFile())
            # 实际上 PaddleOCR 接受文件路径或 numpy array
            # 这里我们将 bytes 保存为临时文件
            import io
            img = Image.open(io.BytesIO(image))
            img_array = np.array(img)

            # 执行 OCR
            result = self.ocr.ocr(img_array, cls=True)

            if result and result[0]:
                for line in result[0]:
                    bbox_points = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    text = line[1][0]  # 识别文本
                    confidence = line[1][1]  # 置信度

                    # 转换四边形为矩形 bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    bbox = (
                        int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)),
                    )

                    # 检测是否为公式区域 (启发式: 含特殊字符比例高)
                    block_type = "text"
                    if self.enable_formula_detection:
                        if self._is_formula_region(text):
                            block_type = "formula"
                            page_result.formulas.append(text)

                    ocr_result = OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        block_type=block_type,
                    )
                    page_result.results.append(ocr_result)

                # 组装全文 (按 y 坐标排序，从上到下)
                page_result.results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
                page_result.full_text = "\n".join(
                    r.text for r in page_result.results
                )

                logger.debug(
                    f"OCR page {page_num}: 识别到 {len(page_result.results)} 个文本块, "
                    f"平均置信度 {sum(r.confidence for r in page_result.results) / len(page_result.results):.2f}"
                )
            else:
                page_result.full_text = ""
                logger.debug(f"OCR page {page_num}: 未识别到文本")

        except Exception as e:
            page_result.success = False
            page_result.error = str(e)
            logger.error(f"OCR 识别失败 (page {page_num}): {e}")

        return page_result

    def process_image_file(
        self, file_path: str, page_num: int = 0
    ) -> OcrPageResult:
        """对图片文件执行 OCR"""
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        return self.process_image(image_bytes, page_num)

    def process_images_batch(
        self, images: List[bytes], start_page: int = 1
    ) -> List[OcrPageResult]:
        """批量处理多张图片 (多页文档)"""
        results = []
        for i, img in enumerate(images):
            result = self.process_image(img, page_num=start_page + i)
            results.append(result)
        return results

    @staticmethod
    def _is_formula_region(text: str) -> bool:
        """
        启发式判断文本是否为公式区域
        特征: 含大量数学符号、希腊字母、上下标
        """
        formula_chars = set(
            "∫∑∏√∞∂∇∈∉⊂⊃∪∩∧∨∀∃αβγδεθλμπστφω"
            "=+-×÷^_{}[]()±≤≥≠≈≡≪≫→⇒⇔°′″"
            "₁₂₃₄₅₆₇₈₉₀⁰¹²³⁴⁵⁶⁷⁸⁹"
        )
        if not text:
            return False
        formula_ratio = sum(1 for c in text if c in formula_chars) / len(text)
        # 如果特殊字符占比超过 15%，判定为公式
        return formula_ratio > 0.15

    def merge_horizontal_texts(
        self, results: List[OCRResult], x_threshold: int = 50
    ) -> List[OCRResult]:
        """
        合并水平相邻的文本块 (同一行拆分的情况)
        """
        if not results:
            return results

        merged = []
        results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))

        current = results[0]
        for next_r in results[1:]:
            # 如果 y 坐标接近 (同一行) 且 x 坐标连续
            y_diff = abs(current.bbox[1] - next_r.bbox[1])
            x_gap = next_r.bbox[0] - current.bbox[2]

            if y_diff < 20 and 0 < x_gap < x_threshold:
                # 合并
                current = OCRResult(
                    text=current.text + next_r.text,
                    confidence=(current.confidence + next_r.confidence) / 2,
                    bbox=(
                        current.bbox[0],
                        min(current.bbox[1], next_r.bbox[1]),
                        next_r.bbox[2],
                        max(current.bbox[3], next_r.bbox[3]),
                    ),
                    block_type=current.block_type,
                )
            else:
                merged.append(current)
                current = next_r

        merged.append(current)
        return merged
