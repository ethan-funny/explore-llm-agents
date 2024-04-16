import ollama
import feedparser
import gradio as gr

from lxml import etree
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def get_content(url):
    # 使用 feedparser 库来解析提供的 URL，通常用于读取 RSS 或 Atom 类型的数据流
    data = feedparser.parse(url)
    
    docs = []
    for news in data['entries']:
        # 通过 xpath 提取干净的文本内容
        summary = etree.HTML(text=news['summary']).xpath('string(.)')
        
        # 初始化文档拆分器，设定块大小和重叠大小
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        
        # 拆分文档
        split_docs = text_splitter.create_documents(texts=[summary], metadatas=[{k: news[k] for k in ('title', 'published', 'link')}])
        
        # 合并文档
        docs.extend(split_docs)

    return data, docs


def create_docs_vector(docs, embeddings):
    # 基于 embeddings，为 docs 创建向量
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def rag_chain(question, vector_store, model='qwen', threshold=0.3):
    # 从向量数据库中检索与 question 相关的文档
    related_docs = vector_store.similarity_search_with_relevance_scores(question)
    
    # 过滤掉小于设定阈值的文档
    related_docs = list(filter(lambda x: x[1] > threshold, related_docs))
    
    # 格式化检索到的文档
    context = "\n\n".join([f'[citation:{i}] {doc[0].page_content}' for i, doc in enumerate(related_docs)])
    
    # 保存文档的 meta 信息，如 title、link 等
    metadata = {str(i): doc[0].metadata for i, doc in enumerate(related_docs)}
    
    # 设定系统提示词
    system_prompt = f"""
    当你收到用户的问题时，请编写清晰、简洁、准确的回答。
    你会收到一组与问题相关的上下文，每个上下文都以参考编号开始，如[citation:x]，其中x是一个数字。
    请使用这些上下文，并在适当的情况下在每个句子的末尾引用上下文。

    你的答案必须是正确的，并且使用公正和专业的语气写作。请限制在1024个tokens之内。
    不要提供与问题无关的信息，也不要重复。
    不允许在答案中添加编造成分，如果给定的上下文没有提供足够的信息，就说“缺乏关于xx的信息”。

    请用参考编号引用上下文，格式为[citation:x]。
    如果一个句子来自多个上下文，请列出所有适用的引用，如[citation:3][citation:5]。
    除了代码和特定的名字和引用，你的答案必须用与问题相同的语言编写，如果问题是中文，则回答也是中文。

    这是一组上下文：

    {context}

    """
    
    user_prompt = f"用户的问题是：{question}"
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': system_prompt + user_prompt
        }
    ])
    
    print(system_prompt + user_prompt)

    return response['message']['content'], context


if __name__ == "__main__":
    hf_embedding = HuggingFaceEmbeddings(model_name="/path/to/bge-m3",
                                         encode_kwargs={'normalize_embeddings': True})
    
    # 财联社 RSS，如果访问不了，可以替换为
    # https://rsshub.feeded.xyz/cls/depth/1003
    url = "https://rsshub.app/cls/depth/1003"
    data, docs = get_content(url)
    vector_store = create_docs_vector(docs, hf_embedding)
    
    # 创建 Gradio 界面
    interface = gr.Interface(
        fn=lambda question, model, threshold: rag_chain(question, vector_store, model, threshold),
        inputs=[
            gr.Textbox(lines=2, placeholder="请输入你的问题...", label="问题"),
            gr.Dropdown(['gemma', 'mistral', 'mixtral', 'qwen:7b'], label="选择模型", value='gemma'),
            gr.Number(label="检索阈值", value=0.3)
        ],
        outputs=[
            gr.Text(label="回答"),
            gr.Text(label="相关上下文")
        ],
        title="资讯问答Bot",
        description="输入问题，我会查找相关资料，然后整合并给你生成回复"
    )

    # 运行界面
    interface.launch()
