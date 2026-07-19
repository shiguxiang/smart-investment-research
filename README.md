# 智能投研分析系统 (Multi-Agent + RAG)

基于多智能体协作的上市公司投研辅助系统，实现从"年报存储"到"智能研判"的闭环。覆盖 500+ 份年报，10+ 行业，万级财务指标索引。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Qwen-Max (DashScope) |
| 多智能体编排 | LangGraph |
| 向量数据库 | Milvus |
| 重排序 | BGE-Reranker-v2-m3 (FlagEmbedding) |
| 缓存 | Redis |
| 评估 | Ragas |
| OCR | PaddleOCR |
| 服务框架 | FastAPI |

## 项目结构

```
smart-review-assistant/
├── config/                     # 全局配置
│   └── settings.py
├── src/
│   ├── main.py                 # FastAPI 入口
│   ├── agents/                 # 多智能体模块
│   │   ├── state.py            # LangGraph 状态定义
│   │   ├── document_agent.py   # 文档解析Agent (版面分析+表格提取+跨页拼接)
│   │   ├── financial_agent.py  # 财务分析Agent (混合检索+口径匹配)
│   │   ├── analysis_agent.py   # 综合研判Agent (投研报告生成)
│   │   └── orchestrator.py     # LangGraph 编排器
│   ├── ingestion/              # 数据摄入模块
│   │   ├── pdf_parser.py       # 年报PDF解析
│   │   ├── ppt_parser.py       # PPT解析
│   │   ├── ocr_processor.py    # OCR处理
│   │   ├── layout_analyzer.py  # 版面分析
│   │   ├── chunker.py          # 文本分块
│   │   └── pipeline.py         # 摄入流水线
│   ├── retrieval/              # 检索模块
│   │   ├── milvus_client.py    # Milvus客户端
│   │   ├── embedder.py         # 向量化
│   │   ├── keyword_search.py   # BM25关键词检索
│   │   ├── vector_search.py    # Milvus向量检索
│   │   ├── hybrid_search.py    # 三路混合检索
│   │   └── reranker.py         # BGE-Reranker重排
│   ├── evaluation/             # 评估模块
│   │   ├── ragas_pipeline.py   # Ragas评估流水线
│   │   └── fallback.py         # 兜底策略
│   ├── api/                    # API层
│   │   ├── routes.py           # REST路由
│   │   └── schemas.py          # 数据模型
│   └── utils/                  # 工具模块
│       ├── logger.py           # 日志
│       └── cache.py            # Redis缓存
├── frontend/                   # 前端 (React + Vite + Tailwind)
│   └── src/
│       ├── App.jsx             # 主应用
│       └── components/         # UI组件
├── scripts/
│   ├── ingest.py               # 批量年报导入
│   └── eval.py                 # 评估脚本
├── tests/                      # 单元测试 (41 cases)
├── data/                       # 原始数据
│   ├── pdfs/                   # 年报PDF
│   ├── ppts/                   # 路演PPT
│   └── exams/                  # 扫描件
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env
# 编辑 .env 填入你的 DASHSCOPE_API_KEY

# 启动 Milvus (本地)
milvus-server

# 启动 Redis
redis-server
```

### 2. 导入年报数据

```bash
# 导入单份年报
python scripts/ingest.py --file "data/pdfs/宁德时代2024年报.pdf" --subject "新能源"

# 批量导入行业目录
python scripts/ingest.py --dir "data/pdfs/银行业/" --subject "银行业" --chapter "2024"
```

### 3. 启动服务

```bash
# 后端
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 前端
cd frontend && npm install && npm run dev
```

访问 http://localhost:5173 使用投研分析界面，或访问 http://localhost:8000/docs 查看 API 文档。

### 4. API 使用

```bash
# 纯文本分析
curl -X POST http://localhost:8000/api/v1/chat \
  -F "query=对比宁德时代和比亚迪2024年营收增速" \
  -F "subject=新能源"

# 带年报分析
curl -X POST http://localhost:8000/api/v1/chat \
  -F "query=这份年报中哪些指标出现异常波动？" \
  -F "subject=银行业" \
  -F "files=@工商银行2024年报.pdf"

# 健康检查
curl http://localhost:8000/api/v1/health

# 触发评估
curl http://localhost:8000/api/v1/eval
```

### 5. 运行测试

```bash
pytest tests/ -v   # 41 tests
```

## 核心架构

### 多智能体编排流程

```
投研分析需求
  │
  ├─ 含年报附件 → 文档解析Agent → 财务分析Agent → 综合研判Agent → 返回
  │                 (版面分析+表格  (三路召回+重排)  (投研报告生成)
  │                  提取+跨页拼接)
  │
  └─ 纯文本需求 → 财务分析Agent → 综合研判Agent → 返回

任一节点异常 → 兜底节点 (fallback)
```

### 三路混合检索

```
分析需求
  │
  ├─ 向量召回 (Milvus + DashScope Embedding)    权重 0.5
  ├─ 关键词召回 (BM25 + jieba分词 + 财务术语扩展)  权重 0.3
  └─ 元数据过滤 (行业/公司/年份)                   权重 0.2
  │
  └─ 加权融合 (Top-20) → BGE-Reranker 精排 → Top-5
```

## 核心指标

| 指标 | 基线 | 优化后 |
|------|------|--------|
| 财务数据提取准确率 | 65% | **88%** |
| Top-5 召回率 | - | **92%** |
| 服务可用性 | - | **97%** |
| 复杂年报处理耗时 | 基准 | **降低 30%** |

## 兜底策略

| 场景 | 策略 |
|------|------|
| 表格解析失败 | 保留原始文本，标注跨页位置供人工核对 |
| LLM 超时 | 返回 Redis 缓存的热门分析结果 |
| Milvus 不可用 | 降级为纯 BM25 关键词检索 |
| 全局异常 | 友好提示 + 日志告警，建议查阅年报原文 |
