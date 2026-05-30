import os
import re
import math
from collections import Counter
from dotenv import load_dotenv
import dashscope
from dashscope import Generation
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

# 参数配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5


def load_textbook(file_path):
    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    full_text = "\n".join(paragraphs)
    return full_text, len(paragraphs)


def split_text(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    return splitter.split_text(text)


def tokenize(text):
    """简单的中文分词（按字和词）"""
    # 提取中文字符串和英文单词
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
    # 额外对中文进行单字分词（捕获单字特征）
    chars = re.findall(r'[\u4e00-\u9fa5]', text)
    return words + chars


def compute_tfidf(chunks):
    """计算所有块的 TF-IDF 向量"""
    # 1. 计算每个块的词频（TF）
    tf_vectors = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        tf = Counter(tokens)
        # 归一化
        total = sum(tf.values())
        if total > 0:
            tf = {k: v / total for k, v in tf.items()}
        tf_vectors.append(tf)

    # 2. 计算逆文档频率（IDF）
    doc_count = len(chunks)
    word_doc_count = {}
    for tf in tf_vectors:
        for word in tf.keys():
            word_doc_count[word] = word_doc_count.get(word, 0) + 1

    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = math.log(doc_count / (count + 1)) + 1

    # 3. 计算 TF-IDF
    tfidf_vectors = []
    for tf in tf_vectors:
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_vectors.append(tfidf)

    return tfidf_vectors, idf


def vector_to_list(vec, vocabulary):
    """将稀疏向量转换为稠密列表（用于余弦相似度计算）"""
    return [vec.get(word, 0) for word in vocabulary]


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def retrieve_context(question, chunks, tfidf_vectors, idf):
    """基于 TF-IDF 的向量检索"""
    # 计算问题的 TF-IDF
    question_tokens = tokenize(question)
    question_tf = Counter(question_tokens)
    total = sum(question_tf.values())
    if total > 0:
        question_tf = {k: v / total for k, v in question_tf.items()}
    else:
        question_tf = {}

    question_tfidf = {word: question_tf.get(word, 0) * idf.get(word, 1) for word in question_tf}

    # 构建词汇表
    all_words = set(question_tfidf.keys())
    for vec in tfidf_vectors[:100]:  # 只取部分来构建词汇表
        all_words.update(vec.keys())
    vocabulary = list(all_words)

    # 转换为向量
    question_vec = vector_to_list(question_tfidf, vocabulary)
    chunk_vecs = [vector_to_list(vec, vocabulary) for vec in tfidf_vectors]

    # 计算相似度
    similarities = []
    for chunk_vec in chunk_vecs:
        sim = cosine_similarity(question_vec, chunk_vec)
        similarities.append(sim)

    # 返回 top_k 个最相似的段落
    indexed = list(enumerate(similarities))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed[:TOP_K]]

    # 打印检索信息（调试用）
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
    file_path = "data/MinerU_docx_人工智能引论_吴飞_解析完成后的教材.docx"

    print("正在读取教材...")
    textbook_text, num_paragraphs = load_textbook(file_path)
    print(f"教材读取完成: {len(textbook_text)} 字符, {num_paragraphs} 段落")

    print("正在分块...")
    chunks = split_text(textbook_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"分块完成: {len(chunks)} 个块")

    print("正在计算 TF-IDF 向量（本地计算，无需网络）...")
    tfidf_vectors, idf = compute_tfidf(chunks)
    print(f"TF-IDF 计算完成，词汇表大小: {len(idf)}")

    print("\n教材助教已启动（TF-IDF 向量检索模式，完全本地）")
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