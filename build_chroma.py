import os
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = 'shibing624/text2vec-base-chinese'  # 中文嵌入模型
PERSIST_DIR = './chroma_db'  # 数据库存储路径


def load_textbook(file_path):
    """读取教材"""
    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    full_text = "\n".join(paragraphs)
    return full_text, len(paragraphs)


def split_text(text, chunk_size, chunk_overlap):
    """文本分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    return splitter.split_text(text)


def build_chroma_database(chunks, embedding_model):
    """构建 ChromaDB 向量数据库"""
    # 初始化 ChromaDB（持久化存储）
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # 删除旧集合（如果存在）
    try:
        client.delete_collection("textbook_collection")
        print("删除旧集合")
    except:
        pass

    # 创建新集合
    collection = client.create_collection(
        name="textbook_collection",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )

    # 批量计算向量并存储
    batch_size = 50
    total = len(chunks)
    print(f"开始向量化 {total} 个文本块...")

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"进度: {i}/{total}")

        # 计算向量
        embeddings = embedding_model.encode(batch).tolist()

        # 生成 ID
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        # 存入 ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=batch,
            ids=ids
        )

    print(f"完成！共存储 {collection.count()} 个向量")
    return collection


if __name__ == "__main__":
    # 1. 读取教材
    file_path = "data/MinerU_docx_人工智能引论_吴飞_解析完成后的教材.docx"
    print("正在读取教材...")
    textbook_text, num_paragraphs = load_textbook(file_path)
    print(f"读取完成: {len(textbook_text)} 字符, {num_paragraphs} 段落")

    # 2. 文本分块
    print("正在分块...")
    chunks = split_text(textbook_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"分块完成: {len(chunks)} 个块")

    # 3. 加载嵌入模型（首次运行会下载，约 80MB）
    print(f"正在加载嵌入模型: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. 构建 ChromaDB
    print("正在构建 ChromaDB...")
    build_chroma_database(chunks, embedding_model)

    print("ChromaDB 构建完成！")