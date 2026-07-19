"""
版面分析模块
识别文档结构：标题、正文、公式、表格区域
将排版信息转化为结构化层级关系
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlockType(Enum):
    """文本块类型"""
    TITLE = "title"           # 标题
    SUBTITLE = "subtitle"     # 副标题/小标题
    PARAGRAPH = "paragraph"   # 正文段落
    FORMULA = "formula"       # 公式
    TABLE = "table"           # 表格
    IMAGE_CAPTION = "image_caption"  # 图注
    LIST_ITEM = "list_item"   # 列表项
    CODE = "code"             # 代码块
    UNKNOWN = "unknown"       # 未分类


@dataclass
class LayoutBlock:
    """版面分析结果块"""
    block_type: BlockType
    text: str
    position: int  # 在原文中的位置 (行号/序号)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["LayoutBlock"] = field(default_factory=list)


@dataclass
class LayoutResult:
    """版面分析结果"""
    blocks: List[LayoutBlock] = field(default_factory=list)
    document_title: str = ""
    sections: List[str] = field(default_factory=list)  # 章节标题列表
    total_blocks: int = 0

    def get_text_by_type(self, block_type: BlockType) -> List[str]:
        """按类型获取文本"""
        return [b.text for b in self.blocks if b.block_type == block_type]

    def to_markdown(self) -> str:
        """转为 Markdown 格式"""
        md_lines = []
        if self.document_title:
            md_lines.append(f"# {self.document_title}\n")

        for block in self.blocks:
            prefix = self._get_markdown_prefix(block.block_type)
            md_lines.append(f"{prefix}{block.text}\n")

        return "\n".join(md_lines)

    @staticmethod
    def _get_markdown_prefix(block_type: BlockType) -> str:
        """获取 Markdown 前缀"""
        mapping = {
            BlockType.TITLE: "# ",
            BlockType.SUBTITLE: "## ",
            BlockType.LIST_ITEM: "- ",
            BlockType.FORMULA: "$$\n",
            BlockType.CODE: "```\n",
            BlockType.PARAGRAPH: "",
            BlockType.TABLE: "",
            BlockType.IMAGE_CAPTION: "> ",
        }
        return mapping.get(block_type, "")


class LayoutAnalyzer:
    """
    版面分析器
    基于启发式规则识别文档结构 (标题/正文/公式/表格)
    """

    # 标题特征模式
    TITLE_PATTERNS = [
        r"^第[一二三四五六七八九十\d]+章",    # 第X章
        r"^第[一二三四五六七八九十\d]+节",    # 第X节
        r"^[一二三四五六七八九十\d]+[、．.]",  # 一、二、三...
        r"^\d+[\.、．]\s",                  # 1. 2. 3.
        r"^[\(（]\d+[\)）]",                 # (1) (2) (3)
        r"^[A-Z][A-Za-z\s]+$",              # 英文大写标题
        r"^[#]{1,3}\s",                     # Markdown 标题
    ]

    # 公式特征
    FORMULA_PATTERNS = [
        r"[∑∫∏√∞∂∇∈∉⊂⊃∪∩∧∨∀∃αβγδεθλμ]",
        r"\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|\\gamma",
        r"=\s*\\begin\{",
        r"f\([xX]\)\s*=",
        r"^\s*[=≠≈±]\s*$",
    ]

    # 列表特征
    LIST_PATTERNS = [
        r"^[\s]*[-•●◆◇▪▸►]\s",
        r"^[\s]*\d+[\.\)]\s",
        r"^[\s]*[（(]\d+[)）]\s",
    ]

    def __init__(
        self,
        min_title_length: int = 2,
        max_title_length: int = 60,
        min_paragraph_length: int = 10,
    ):
        self.min_title_length = min_title_length
        self.max_title_length = max_title_length
        self.min_paragraph_length = min_paragraph_length

    def analyze(self, text: str, source: str = "") -> LayoutResult:
        """
        分析文本版面结构

        Args:
            text: 输入文本 (可以是单个段落或多页拼接)
            source: 来源标识 (文件名等)

        Returns:
            LayoutResult 包含结构化的版面块列表
        """
        result = LayoutResult()
        lines = text.strip().split("\n")

        # 过滤空行
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        if not non_empty_lines:
            return result

        # 尝试提取文档标题 (第一行，如果够短)
        if len(non_empty_lines[0]) <= self.max_title_length:
            result.document_title = non_empty_lines[0]

        for i, line in enumerate(non_empty_lines):
            block_type = self._classify_line(line)

            # 记录章节
            if block_type == BlockType.TITLE:
                result.sections.append(line)
            elif block_type == BlockType.SUBTITLE:
                result.sections.append(line)

            block = LayoutBlock(
                block_type=block_type,
                text=line,
                position=i,
                metadata={"source": source, "line_index": i},
            )
            result.blocks.append(block)

        result.total_blocks = len(result.blocks)

        logger.debug(
            f"版面分析完成: 共 {result.total_blocks} 个块, "
            f"标题 {len(result.sections)} 个, "
            f"来源: {source}"
        )

        return result

    def _classify_line(self, line: str) -> BlockType:
        """分类单行文本类型"""
        if not line:
            return BlockType.UNKNOWN

        # 1. 检查是否为公式
        if self._is_formula(line):
            return BlockType.FORMULA

        # 2. 检查是否为代码块
        if self._is_code(line):
            return BlockType.CODE

        # 3. 检查是否为列表项
        if self._is_list_item(line):
            return BlockType.LIST_ITEM

        # 4. 检查是否为表格分隔行
        if self._is_table_row(line):
            return BlockType.TABLE

        # 5. 检查是否为图注
        if self._is_image_caption(line):
            return BlockType.IMAGE_CAPTION

        # 6. 检查是否为标题 (长度优先判断)
        if self.min_title_length <= len(line) <= self.max_title_length:
            if self._is_title(line):
                if len(line) <= 20:
                    return BlockType.TITLE
                return BlockType.SUBTITLE

        # 7. 默认为正文段落
        return BlockType.PARAGRAPH

    def _is_title(self, line: str) -> bool:
        """判断是否为标题"""
        for pattern in self.TITLE_PATTERNS:
            if re.match(pattern, line):
                return True

        # 短文本 + 无标点结尾 → 可能是标题
        if len(line) <= 25 and not re.search(r'[。，；：""''）】)]$', line):
            return True

        return False

    def _is_formula(self, line: str) -> bool:
        """判断是否为公式"""
        # LaTeX 特征
        if line.count("$") >= 2:
            return True
        if "\\begin{" in line or "\\end{" in line:
            return True

        # 数学符号高密度
        math_chars = len(re.findall(r"[=+\-*/^_{}\[\]()∂∫∑∏√αβγδεθλμπστφω]", line))
        alpha_chars = len(re.findall(r"[a-zA-Z一-鿿]", line))
        if alpha_chars > 0 and math_chars / (alpha_chars + math_chars) > 0.3:
            return True

        return False

    def _is_code(self, line: str) -> bool:
        """判断是否为代码"""
        code_indicators = [
            r"^\s*def\s+\w+\(",    # Python 函数
            r"^\s*class\s+\w+",    # Python 类
            r"^\s*import\s+",      # import 语句
            r"^\s*from\s+\w+\s+import",
            r"^\s*\{\s*$",         # JSON/C 风格
            r"^\s*\}\s*$",
            r"^\s*<\w+>",          # HTML/XML 标签
        ]
        for pattern in code_indicators:
            if re.match(pattern, line):
                return True
        return False

    def _is_list_item(self, line: str) -> bool:
        """判断是否为列表项"""
        for pattern in self.LIST_PATTERNS:
            if re.match(pattern, line):
                return True
        return False

    def _is_table_row(self, line: str) -> bool:
        """判断是否为表格行"""
        # 含多个 | 分隔符或制表符
        if line.count("|") >= 2:
            return True
        if "\t" in line:
            return True
        return False

    def _is_image_caption(self, line: str) -> bool:
        """判断是否为图注/表注"""
        caption_patterns = [
            r"^图\d+[\.:：\s]",
            r"^表\d+[\.:：\s]",
            r"^Fig(ure)?\.?\s*\d+",
            r"^Table\.?\s*\d+",
        ]
        for pattern in caption_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False

    def analyze_document_structure(
        self, text: str, source: str = ""
    ) -> List[Dict]:
        """
        分析文档的层级结构 (返回章节树)

        Returns:
            [{title, level, content, children}] 格式的章节树
        """
        result = self.analyze(text, source)
        sections = []
        current_section = None

        for block in result.blocks:
            if block.block_type in (BlockType.TITLE, BlockType.SUBTITLE):
                level = 1 if block.block_type == BlockType.TITLE else 2
                current_section = {
                    "title": block.text,
                    "level": level,
                    "content": [],
                    "children": [],
                }
                sections.append(current_section)
            elif current_section is not None:
                current_section["content"].append(block.text)

        return sections
