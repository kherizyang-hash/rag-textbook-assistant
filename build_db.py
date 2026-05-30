import pickle
import os
import re
import math
from collections import Counter
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 配置 ====================
#CHUNK_SIZE = 800
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


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
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
    chars = re.findall(r'[\u4e00-\u9fa5]', text)
    return words + chars


def compute_tfidf(chunks):
    # TF
    tf_vectors = []
    for chunk in chunks:
        tokens = tokenize(chunk)
        tf = Counter(tokens)
        total = sum(tf.values())
        if total > 0:
            tf = {k: v / total for k, v in tf.items()}
        tf_vectors.append(tf)

    # IDF
    doc_count = len(chunks)
    word_doc_count = {}
    for tf in tf_vectors:
        for word in tf.keys():
            word_doc_count[word] = word_doc_count.get(word, 0) + 1

    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = math.log(doc_count / (count + 1)) + 1

    # TF-IDF
    tfidf_vectors = []
    for tf in tf_vectors:
        tfidf = {word: tf[word] * idf[word] for word in tf}
        tfidf_vectors.append(tfidf)

    return tfidf_vectors, idf


if __name__ == "__main__":
    file_path = "data/MinerU_docx_人工智能引论_吴飞_解析完成后的教材.docx"

    print("正在读取教材...")
    textbook_text, num_paragraphs = load_textbook(file_path)
    print(f"教材读取完成: {len(textbook_text)} 字符, {num_paragraphs} 段落")

    print("正在分块...")
    chunks = split_text(textbook_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"分块完成: {len(chunks)} 个块")

    print("正在计算 TF-IDF...")
    tfidf_vectors, idf = compute_tfidf(chunks)
    print(f"词汇表大小: {len(idf)}")

    print("正在保存知识库...")
    knowledge_base = {
        'chunks': chunks,
        'tfidf_vectors': tfidf_vectors,
        'idf': idf,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP
    }
    with open('knowledge_base_backup.pkl', 'wb') as f:
        pickle.dump(knowledge_base, f)
    print("知识库已保存到 knowledge_base_backup.pkl")