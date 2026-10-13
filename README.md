# rag-chat · 智能知识库问答系统（RAG）

基于 Python 的检索增强生成（RAG）问答系统：文档 → 切块 → 向量化 → 相似度检索 → 带上下文生成。

## 技术栈
Python · OpenAI 兼容 API · NumPy（轻量向量索引）· FastAPI · SSE 流式输出

## 快速开始
```bash
# 1. 配置 API Key（推荐硅基流动，免费）
copy .env.example .env        # Windows；Linux/macOS 用 cp
# 编辑 .env 填入 OPENAI_API_KEY

# 2. 安装依赖
pip install -r requirements.txt

# 3. 构建向量索引
python ingest.py

# 4. 命令行问答
python query.py

# 5. 启动 Web 接口（可选）
uvicorn app:app --reload
```

## 接口
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/ask | 普通问答，返回完整回答 |
| GET | /api/ask/stream?question=xxx | SSE 流式问答 |

## 说明
- `data/sample.md` 为示例知识库，可替换为任意 .md/.txt 文档。
- 向量检索用 NumPy 余弦相似度实现（可替换为 Chroma/Milvus 等向量数据库）。