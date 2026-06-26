import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

import gradio as gr

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "BAAI/bge-small-en-v1.5"


def get_embedding_model_config() -> tuple[str, dict]:
    """Prefer a local SentenceTransformer cache, and fall back to Hugging Face download."""
    env_model_path = os.getenv("BOOK_EMBEDDING_MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path).expanduser()
        if model_path.exists():
            return str(model_path), {"local_files_only": True}
        raise FileNotFoundError(
            "BOOK_EMBEDDING_MODEL_PATH 指向的目录不存在，请检查 .env 中的路径。"
        )

    local_models = [
        BASE_DIR / "models" / "bge-small-en-v1.5",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5",
    ]
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        local_models.append(Path(hf_home).expanduser() / "hub" / "models--BAAI--bge-small-en-v1.5")

    for model_dir in local_models:
        snapshots_dir = model_dir / "snapshots"
        if model_dir.is_dir() and (model_dir / "modules.json").exists():
            return str(model_dir)
        if snapshots_dir.is_dir():
            snapshots = sorted(
                snapshots_dir.iterdir(),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            for snapshot in snapshots:
                if (snapshot / "modules.json").exists():
                    return str(snapshot), {"local_files_only": True}

    return MODEL_ID, {}


# --- 1. 加载 CSV 数据 ---
books = pd.read_csv(BASE_DIR / "books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    str(BASE_DIR / "cover-not-found.jpg"),
    books["large_thumbnail"],
)

# --- 2. 加载持久化的向量数据库 (高效模式) ---
embedding_model_name, embedding_model_kwargs = get_embedding_model_config()
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
)

# 注意：这里不是 from_documents，而是直接加载
chroma_dir = BASE_DIR / "chroma_db"
if chroma_dir.exists():
    db_books = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
else:
    # 兼容你还没运行第一步的情况，但强烈建议先运行第一步
    print("Warning: Persisted DB not found. Building in memory (slow)...")
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter

    raw_documents = TextLoader(str(BASE_DIR / "tagged_description.txt"), encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embedding=embeddings)

# --- 3. 配置 DeepSeek ---
def build_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.7,
    )


llm = build_llm()


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 100,
        final_top_k: int = 16,
) -> pd.DataFrame:
    category = category or "All"
    tone = tone or "All"

    # 向量检索
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # 类别过滤
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # 情感排序
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query, category, tone):
    if not query or not query.strip():
        return "请输入搜索描述。", [], pd.DataFrame()

    # 1. 获取基础推荐数据
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    # 如果没有结果，返回空
    if recommendations.empty:
        return "未找到相关书籍。", [], pd.DataFrame()

    # 2. 准备 DeepSeek 的输入 (取前 5 本书的信息)
    top_books = recommendations.head(5)
    books_info = ""
    for _, row in top_books.iterrows():
        books_info += f"- 《{row['title']}》 (Category: {row['simple_categories']})\n"

    # 3. 让 DeepSeek 生成推荐语
    prompt = f"""
    用户正在寻找 "{query}" 类型的书，基调倾向 "{tone}"。
    检索到的书籍：
    {books_info}
    请以专业图书管理员的身份（中文），根据用户意图总结推荐理由。
    语气专业且亲切，字数控制在 300 字以内。
    """
    if llm is None:
        ai_comment = "已根据语义相似度返回推荐结果。配置 DEEPSEEK_API_KEY 后，可生成中文推荐理由。"
    else:
        try:
            response = llm.invoke(prompt)
            ai_comment = response.content
        except Exception as e:
            ai_comment = f"搜索完成。无法生成 AI 评论: {str(e)}"

    # 4. 格式化 Gallery 输出 (只保留封面和书名，保持界面清爽)
    results = []
    for _, row in recommendations.iterrows():
        # 截断作者名以防止过长
        authors = row["authors"].split(";")[0]
        caption = f"《{row['title']}》\n{authors}"
        results.append((row["large_thumbnail"], caption))

    # 返回: AI评论, 画廊数据列表, 完整的DataFrame(用于后续点击查看详情)
    return ai_comment, results, recommendations


def on_select_book(evt: gr.SelectData, df_recs):
    if df_recs is None or df_recs.empty:
        return "请先搜索书籍"

    # 获取用户点击的索引
    index = evt.index
    if index < len(df_recs):
        row = df_recs.iloc[index]

        # 格式化详情文本
        details = f"""
        ### {row['title']}
        
        **作者**: {row['authors'].replace(';', ', ')}
        **类别**: {row['simple_categories']}
        
        **简介**:
        {row['description']}
        """
        return details
    return "无法获取书籍信息"


# --- Gradio UI 布局调整 ---
categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(title="智能图书推荐") as dashboard:
    # 状态变量，用于存储当前的搜索结果数据
    current_books_state = gr.State()

    gr.Markdown("# 智能图书推荐系统")

    with gr.Row():
        with gr.Column(scale=1):
            user_query = gr.Textbox(
                label="搜索描述",
                placeholder="请输入英文描述，例如: A story about forgiveness",
                lines=2
            )
            with gr.Row():
                category_dropdown = gr.Dropdown(choices=categories, label="书籍类别", value="All")
                tone_dropdown = gr.Dropdown(choices=tones, label="情感基调", value="All")

            submit_button = gr.Button("开始检索", variant="primary")

            gr.Markdown("### AI 推荐助手")
            ai_output = gr.Markdown(value="请输入描述并点击检索，AI 将为您分析推荐理由。")

        with gr.Column(scale=2):
            gr.Markdown("### 推荐书单")
            # gallery output
            gallery_view = gr.Gallery(
                show_label=False,
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                allow_preview=False  # 关闭默认的预览弹窗，改为自定义详情展示
            )

            # 书籍详情展示区
            gr.Markdown("### 书籍详情")
            book_detail_view = gr.Markdown(
                value="请点击上方任意一本书籍封面，此处将显示完整简介。",
                elem_classes="book-details"  # 可用于自定义CSS（可选）
            )

    # 事件绑定
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[ai_output, gallery_view, current_books_state]  # 更新 UI 和 状态
    )

    # 绑定 Gallery 的选择事件
    gallery_view.select(
        fn=on_select_book,
        inputs=[current_books_state],  # 传入当前存储的书籍数据
        outputs=[book_detail_view]  # 输出到详情框
    )

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())
