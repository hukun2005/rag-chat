"""核心 RAG：文本切块 + 向量检索 + 提示词构建"""
import json
import re
import numpy as np


def split_text(text, chunk_size=400, overlap=60):
    """简单切块：优先按段落，超长段落再按句子滑动切分"""
    text = text.strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= chunk_size:
            chunks.append(p)
            continue
        sentences = re.split(r"(?<=[。！？；\n])", p)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) > chunk_size and buf:
                chunks.append(buf)
                buf = buf[-overlap:] + s if overlap else s
            else:
                buf += s
        if buf.strip():
            chunks.append(buf)
    return chunks


class VectorStore:
    """基于 NumPy 的轻量向量索引（余弦相似度），持久化到 JSON"""

    def __init__(self, path="index.json"):
        self.path = path
        self.chunks = []
        self.vectors = []

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "vectors": self.vectors}, f, ensure_ascii=False)

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            d = json.load(f)
            self.chunks = d["chunks"]
            self.vectors = d["vectors"]

    def search(self, query_vec, top_k=4):
        """返回 [(相似度, 文本), ...]，按相似度降序"""
        if not self.vectors:
            return []
        mat = np.asarray(self.vectors, dtype=float)
        q = np.asarray(query_vec, dtype=float)
        q = q / (np.linalg.norm(q) + 1e-9)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ q
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.chunks[i]) for i in idx]


SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请仅依据下面的【参考上下文】回答用户问题，"
    "上下文没有的信息不要编造；若无法回答，直接说明“知识库中暂无相关信息”。\n\n"
    "【参考上下文】\n{context}"
)


def build_messages(question, context):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]