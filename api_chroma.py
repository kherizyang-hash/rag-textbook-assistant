from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

load_dotenv()

app = FastAPI()

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = 'shibing624/text2vec-base-chinese'
TOP_K = 5

# 加载嵌入模型
print("正在加载嵌入模型...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("嵌入模型加载完成")

# 加载 ChromaDB
print("正在加载 ChromaDB...")
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection("textbook_collection")
print(f"ChromaDB 加载完成，共 {collection.count()} 个向量")


def retrieve_context(question, top_k=TOP_K):
    """使用 ChromaDB 检索相关段落"""
    # 将问题转换为向量
    question_embedding = embedding_model.encode([question]).tolist()

    # 检索
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    return results['documents'][0]


def generate_answer(question, context):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    prompt = f"""你是教材的智能助教。你的任务是根据【教材内容】回答问题。

【教材内容】
{context}

【问题】
{question}

【要求】
1. 严格基于【教材内容】回答，不能编造教材中没有的内容
2. 用自己的话重新组织，不要逐字复制教材原文
3. 如果相关内容在教材中分布在多处，请综合归纳
4. 回答要有条理，可以使用列表或分段
5. 如果教材内容不足以回答问题，请明确说“教材中未详细说明”
6. 回答要像老师在讲课，而不是在念书
7. 按逻辑顺序组织成简洁的回答

【回答】"""

    response = Generation.call(
        model='qwen-plus',
        prompt=prompt,
        result_format='message'
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"调用失败: {response.message}"


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question
    print(f"收到问题: {question}")

    # 检索相关段落
    context_chunks = retrieve_context(question)
    context = "\n\n---\n\n".join(context_chunks)

    # 生成回答
    answer = generate_answer(question, context)

    return {"answer": answer, "sources": context_chunks}


@app.get("/")
async def root():
    return {"message": "教材助教 API 正在运行（ChromaDB 版本）", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)