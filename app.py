"""FastAPI 接口：POST /api/ask 与 SSE 流式 /api/ask/stream"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import VectorStore, build_messages
from llm import embed_texts, chat_stream

INDEX_PATH = "index.json"

store = VectorStore(INDEX_PATH)
store.load()

app = FastAPI(title="RAG 知识库问答")


class Question(BaseModel):
    question: str


def retrieve(question, top_k=4):
    vec = embed_texts([question])[0]
    hits = store.search(vec, top_k=top_k)
    return "\n\n".join(t for _, t in hits)


@app.post("/api/ask")
def ask(q: Question):
    answer = "".join(chat_stream(build_messages(q.question, retrieve(q.question))))
    return {"answer": answer}


@app.get("/api/ask/stream")
def ask_stream(question: str):
    def gen():
        for delta in chat_stream(build_messages(question, retrieve(question))):
            yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")