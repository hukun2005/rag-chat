"""命令行交互式问答"""
from rag import VectorStore, build_messages
from llm import embed_texts, chat_stream

INDEX_PATH = "index.json"


def main():
    store = VectorStore(INDEX_PATH)
    store.load()
    print(f"知识库已加载（{len(store.chunks)} 个片段）。输入问题开始提问，输入 exit 退出。")
    while True:
        q = input("\n你：").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        vec = embed_texts([q])[0]
        hits = store.search(vec, top_k=4)
        context = "\n\n".join(t for _, t in hits)
        print(f"[检索到 {len(hits)} 条相关片段]")
        print("助手：", end="", flush=True)
        for delta in chat_stream(build_messages(q, context)):
            print(delta, end="", flush=True)
        print()


if __name__ == "__main__":
    main()