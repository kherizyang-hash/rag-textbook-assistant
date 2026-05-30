import os
import re
from dotenv import load_dotenv
import dashscope
from dashscope import Generation
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = api_key

# 实验参数配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
TOP_K = 5

def load_textbook(file_path):
    """读取教材文件"""
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


def keyword_retrieve(question, chunks, TOP_K):
    """
    基于关键词检索相关段落
    提取问题中的关键词（去掉停用词），在 chunks 中匹配
    """
    # 简单的关键词提取（取长度>=2的词语）
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', question)
    keywords = [w for w in words if len(w) >= 2]

    # 计算每个 chunk 的关键词匹配分数
    scores = []
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw in chunk)
        scores.append(score)

    # 返回得分最高的 TOP_K 个 chunks
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in indexed[:TOP_K] if score > 0]

    # 如果没匹配到任何关键词，返回前 TOP_K 个 chunks
    if not top_indices:
        return chunks[:TOP_K]

    print(f"检索到的关键词: {keywords}")
    for i, idx in enumerate(top_indices[:TOP_K]):
        print(f"段落{i + 1}: {chunks[idx][:100]}...")

    return [chunks[i] for i in top_indices]


def generate_answer(question, context):
    """调用 Qwen 生成回答"""
    prompt = f"""你是基于教材的智能助教。请根据以下教材内容回答问题。

【教材内容】
{context}

【问题】
{question}

【要求】
- 只使用上述教材中包含的内容回答
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


def ask(question, chunks, TOP_K):
    """完整的问答流程"""
    context_chunks = keyword_retrieve(question, chunks, TOP_K)
    context = "\n\n---\n\n".join(context_chunks)
    answer = generate_answer(question, context)
    return answer, context_chunks


if __name__ == "__main__":
    file_path = "data/MinerU_docx_人工智能引论_吴飞_解析完成后的教材.docx"

    print("正在读取教材...")
    textbook_text, num_paragraphs = load_textbook(file_path)
    print(f"教材读取完成: {len(textbook_text)} 字符, {num_paragraphs} 段落")

    print("正在分块...")
    chunks = split_text(textbook_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"分块完成: {len(chunks)} 个块")

    print("\n教材助教已启动（关键词检索模式）")
    print("输入问题开始提问，输入 exit 退出\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("正在检索...")
        answer, used_chunks = ask(question, chunks,TOP_K)
        print(f"回答: {answer}")
        print(f"(检索到 {len(used_chunks)} 个相关段落)\n")