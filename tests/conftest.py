"""
Pytest 配置文件
提供测试 fixtures
"""

import os
import sys
import pytest

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_pdf_text():
    """示例 PDF 文本"""
    return """
第一章 极限与连续

1.1 极限的定义

设函数 f(x) 在 x₀ 的某个去心邻域内有定义。如果存在常数 A，
对于任意给定的 ε > 0，总存在 δ > 0，使得当 0 < |x - x₀| < δ 时，
有 |f(x) - A| < ε，则称 A 为 f(x) 在 x → x₀ 时的极限。

记作: lim(x→x₀) f(x) = A

1.2 极限的性质

1. 唯一性: 若极限存在，则极限值唯一。
2. 局部有界性: 若极限存在，则函数在 x₀ 附近有界。
3. 保号性: 若极限 A > 0，则在 x₀ 附近 f(x) > 0。
"""


@pytest.fixture
def sample_query():
    """示例查询"""
    return "请解释极限的定义"


@pytest.fixture
def sample_chunks():
    """示例文本块"""
    return [
        type("Chunk", (), {
            "chunk_id": "chunk_00001",
            "text": "极限定义为设函数f(x)在x₀的某个去心邻域内有定义...",
            "metadata": {"subject": "高等数学", "chapter": "第一章"},
            "chunk_index": 0,
        }),
        type("Chunk", (), {
            "chunk_id": "chunk_00002",
            "text": "极限具有唯一性、局部有界性和保号性三个重要性质...",
            "metadata": {"subject": "高等数学", "chapter": "第一章"},
            "chunk_index": 1,
        }),
    ]
