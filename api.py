from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import math
from collections import Counter
import os
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

load_dotenv()

app = FastAPI()

# 允许前端访问（解决跨域问题）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载知识库
print("正在加载知识库...")
with open('knowledge_base_backup.pkl', 'rb') as f:
    kb = pickle.load(f)

chunks = kb['chunks']
tfidf_vectors = kb['tfidf_vectors']
idf = kb['idf']
print(f"知识库加载完成，共 {len(chunks)} 个块")

TOP_K = 5


def tokenize(text):
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
    chars = re.findall(r'[\u4e00-\u9fa5]', text)
    return words + chars


def vector_to_list(vec, vocabulary):
    return [vec.get(word, 0) for word in vocabulary]


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def retrieve_context(question, top_k=5, min_similarity=0.1):
    # 问题的 TF-IDF
    question_tokens = tokenize(question)
    question_tf = {}
    for token in question_tokens:
        question_tf[token] = question_tf.get(token, 0) + 1
    total = sum(question_tf.values())
    if total > 0:
        question_tf = {k: v / total for k, v in question_tf.items()}

    question_tfidf = {word: question_tf.get(word, 0) * idf.get(word, 1) for word in question_tf}

    # 构建词汇表
    all_words = set(question_tfidf.keys())
    for vec in tfidf_vectors[:100]:
        all_words.update(vec.keys())
    vocabulary = list(all_words)

    question_vec = vector_to_list(question_tfidf, vocabulary)
    chunk_vecs = [vector_to_list(vec, vocabulary) for vec in tfidf_vectors]

    similarities = [cosine_similarity(question_vec, chunk_vec) for chunk_vec in chunk_vecs]

    # 自适应：只取相似度 > min_similarity 的段落，最多 top_k 个
    valid_indices = [(i, s) for i, s in enumerate(similarities) if s > min_similarity]
    valid_indices.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, s in valid_indices[:top_k]]

    # 如果没找到任何相关段落，返回相似度最高的前 top_k 个（保底）
    if not top_indices:
        indexed = list(enumerate(similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed[:top_k]]

    return [chunks[idx] for idx in top_indices]


def generate_answer(question, context):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    prompt = f"""你是基于教材的智能助教。请根据以下教材内容回答问题。

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


class AnswerResponse(BaseModel):
    answer: str
    sources: list


@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question

    print(f"收到问题: {question}")

    # 检索相关段落
    context_chunks = retrieve_context(question, top_k=5, min_similarity=0.08)
    context = "\n\n---\n\n".join(context_chunks)

    # 生成回答
    answer = generate_answer(question, context)

    return {"answer": answer, "sources": context_chunks}


@app.get("/")
async def root():
    return {"message": "教材助教 API 正在运行", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)