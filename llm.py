"""LLM 与 Embedding 客户端封装（OpenAI 兼容接口，默认硅基流动）"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

if not API_KEY:
    raise RuntimeError("未配置 OPENAI_API_KEY，请复制 .env.example 为 .env 并填入 API Key")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60)


def embed_texts(texts):
    """将文本列表转为向量列表"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def chat(messages, stream=False, temperature=0.7):
    """对话补全（可选流式）"""
    return client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, stream=stream, temperature=temperature
    )


def chat_stream(messages):
    """流式对话生成器，逐段 yield 文本"""
    for chunk in chat(messages, stream=True):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta