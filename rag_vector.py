import os
from dotenv import load_dotenv
import dashscope
from dashscope import Generation
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib

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


def build_vector_database(chunks, embedding_model, persist_dir="./chroma_db"):
    """将文本块向量化并存储到 ChromaDB"""
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("textbook_collection")
    except:
        pass
    collection = client.create_collection(name="textbook_collection")

    batch_size = 50
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"向量化进度: {i}/{total}")
        embeddings = embedding_model.encode(batch).tolist()
        ids = [hashlib.md5(chunk.encode()).hexdigest()[:16] for chunk in batch]
        collection.add(embeddings=embeddings, documents=batch, ids=ids)

    return collection


def retrieve_context(question, collection, embedding_model, top_k=TOP_K):
    """向量检索"""
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    return results['documents'][0]


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

    print("正在加载嵌入模型（首次运行会下载约80MB）...")
    embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
    print("嵌入模型加载完成")

    print("正在构建向量数据库...")
    collection = build_vector_database(chunks, embedding_model)
    print(f"向量数据库构建完成，共 {collection.count()} 条记录")

    print("\n教材助教已启动（向量检索模式）")
    print("输入问题开始提问，输入 exit 退出\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("正在检索...")
        context_chunks = retrieve_context(question, collection, embedding_model)
        context = "\n\n---\n\n".join(context_chunks)
        answer = generate_answer(question, context)
        print(f"回答: {answer}")
        print(f"(检索到 {len(context_chunks)} 个相关段落)\n")