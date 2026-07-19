"""
LangGraph Agent 状态定义
定义 Multi-Agent 协作中的共享状态结构
"""

from typing import List, Dict, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
import operator


@dataclass
class DocRef:
    """文档引用"""
    doc_id: str
    text: str
    score: float = 0.0
    file_name: str = ""
    subject: str = ""
    chapter: str = ""
    chunk_index: int = 0

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "score": self.score,
            "file_name": self.file_name,
            "subject": self.subject,
            "chapter": self.chapter,
        }


class AgentState(TypedDict):
    """
    LangGraph 共享状态

    各节点通过读取/更新此状态来完成协作任务流转
    """
    # ===== 输入 =====
    query: str                             # 用户原始问题
    files: List[str]                       # 待处理文件路径列表
    subject: str                           # 科目 (可选)
    session_id: str                        # 会话 ID

    # ===== OCR 解析结果 =====
    ocr_text: str                          # OCR 提取的完整文本
    ocr_success: bool                      # OCR 是否成功
    ocr_error: Optional[str]               # OCR 错误信息
    ocr_stats: Dict[str, Any]              # OCR 统计信息

    # ===== 检索结果 =====
    retrieved_docs: List[Dict]             # 向量检索结果
    keyword_docs: List[Dict]               # 关键词检索结果
    hybrid_results: List[Dict]             # 混合检索 + 重排后的最终结果
    retrieval_success: bool                # 检索是否成功

    # ===== 推理结果 =====
    final_answer: str                      # 最终回答
    references: List[Dict]                 # 引用来源
    reasoning_trace: str                   # 推理过程 (chain-of-thought)

    # ===== 流程控制 =====
    error: Optional[str]                   # 全局错误信息 (触发兜底)
    fallback_used: bool                    # 是否使用了兜底策略
    current_step: str                      # 当前执行步骤 (用于监控)
    history: Annotated[List[Dict], operator.add]  # 对话历史 (追加)
