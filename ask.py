import pickle
import math
import re
import os
from dotenv import load_dotenv
import dashscope
from dashscope import Generation

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

#TOP_K = 5
TOP_K = 8

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


def retrieve_context(question, chunks, tfidf_vectors, idf):
    # 计算问题的 TF-IDF
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

    # 转换为向量
    question_vec = vector_to_list(question_tfidf, vocabulary)
    chunk_vecs = [vector_to_list(vec, vocabulary) for vec in tfidf_vectors]

    # 计算相似度
    similarities = [cosine_similarity(question_vec, chunk_vec) for chunk_vec in chunk_vecs]

    # 返回 top_k 个
    indexed = list(enumerate(similarities))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:TOP_K]]

    print(f"检索到的词: {list(question_tfidf.keys())[:10]}")
    for i, idx in enumerate(top_indices[:TOP_K]):
        preview = chunks[idx][:80].replace("\n", " ")
        print(f"段落{i + 1} (相似度={similarities[idx]:.3f}): {preview}...")

    return [chunks[idx] for idx in top_indices]


def generate_answer(question, context):
    prompt = f"""你是基于教材的智能助教。请根据以下教材内容回答问题。

【教材内容】
{context}

【问题】
{question}

【要求】
- 只使用上述教材内容回答
- 如果教材中没有相关信息，请明确说明
- 回答简洁准确

【回答】"""

    response = Generation.call(
        model='qwen-turbo',
        prompt=prompt,
        result_format='message'
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"调用失败: {response.message}"


if __name__ == "__main__":
    # 加载知识库
    print("正在加载知识库...")
    with open('knowledge_base_backup.pkl', 'rb') as f:
        kb = pickle.load(f)

    chunks = kb['chunks']
    tfidf_vectors = kb['tfidf_vectors']
    idf = kb['idf']
    print(f"知识库加载完成，共 {len(chunks)} 个块")

    print("\n教材助教已启动（TF-IDF 向量检索模式）")
    print("输入问题开始提问，输入 exit 退出\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("正在检索...")
        context_chunks = retrieve_context(question, chunks, tfidf_vectors, idf)
        context = "\n\n---\n\n".join(context_chunks)
        answer = generate_answer(question, context)

        print(f"回答: {answer}")
        print(f"(检索到 {len(context_chunks)} 个相关段落)\n")