"""将 data 目录下的文档向量化并建立索引"""
import os
import glob
from rag import split_text, VectorStore
from llm import embed_texts

DATA_DIR = "data"
INDEX_PATH = "index.json"


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.md")) + glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print(f"{DATA_DIR} 目录下没有文档，请放入 .md 或 .txt 文件")
        return

    store = VectorStore(INDEX_PATH)
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_text(text)
        print(f"[{fp}] 切分为 {len(chunks)} 个片段，开始向量化...")
        store.add(chunks, embed_texts(chunks))

    store.save()
    print(f"索引构建完成，共 {len(store.chunks)} 个片段，已保存到 {INDEX_PATH}")


if __name__ == "__main__":
    main()