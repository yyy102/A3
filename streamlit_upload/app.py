import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP 语义表示方法交互系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Shadcn-inspired CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --background: #09090b;
    --foreground: #fafafa;
    --card: #18181b;
    --card-foreground: #fafafa;
    --border: #27272a;
    --input: #27272a;
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --muted: #27272a;
    --muted-foreground: #a1a1aa;
    --accent: #1e1e2e;
    --success: #22c55e;
    --warning: #f59e0b;
    --destructive: #ef4444;
    --radius: 0.5rem;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--background) !important;
    color: var(--foreground) !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem;
    max-width: 1400px;
}

/* Header */
.nlp-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.nlp-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.nlp-header p {
    color: var(--muted-foreground);
    font-size: 0.95rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--muted) !important;
    border-radius: var(--radius) !important;
    padding: 3px !important;
    gap: 2px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--muted-foreground) !important;
    border-radius: calc(var(--radius) - 2px) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--card) !important;
    color: var(--foreground) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--muted-foreground);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 9999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem;
}
.badge-primary { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-success { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-warning { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-danger  { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-info    { background: rgba(6,182,212,0.15);   color: #22d3ee; border: 1px solid rgba(6,182,212,0.3); }

/* Result rows */
.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-radius: var(--radius);
    margin-bottom: 0.35rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    transition: background 0.15s;
}
.result-row:hover { background: rgba(255,255,255,0.06); }
.result-word { font-weight: 600; font-size: 0.9rem; }
.result-score { font-size: 0.8rem; color: var(--muted-foreground); }
.score-bar-bg { flex:1; height: 4px; background: var(--muted); border-radius: 2px; margin: 0 0.75rem; }
.score-bar    { height: 4px; border-radius: 2px; background: linear-gradient(90deg, #6366f1, #a78bfa); }

/* Info / Alert boxes */
.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #c4b5fd;
    line-height: 1.6;
}
.success-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #4ade80;
}
.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #f87171;
}
.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: var(--radius);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #fbbf24;
}

/* Metric cards */
.metric-grid { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
}
.metric-label { font-size: 0.75rem; color: var(--muted-foreground); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: var(--foreground); margin: 0.2rem 0; }
.metric-sub   { font-size: 0.75rem; color: var(--muted-foreground); }

/* Section divider */
.section-divider {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.25rem 0;
}
.section-divider .line { flex:1; height:1px; background: var(--border); }
.section-divider .label { font-size: 0.75rem; color: var(--muted-foreground); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; }

/* Streamlit widget tweaks */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--foreground) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--primary-hover) !important; }
.stSlider > div { color: var(--foreground) !important; }

/* Radio */
.stRadio > div { gap: 0.5rem; }
.stRadio [data-baseweb="radio"] { background: var(--card) !important; }

label, .stLabel { color: var(--muted-foreground) !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Default corpus — The Little Prince excerpt
# ─────────────────────────────────────────────
DEFAULT_CORPUS = """Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest.
It was a picture of a boa constrictor in the act of swallowing an animal.
In the book it said: Boa constrictors swallow their prey whole, without chewing it.
After that I never used to say: Give me that, I want it.
I used to say: Draw me a sheep.
The grown-ups never understand anything by themselves, and it is tiresome for children to be always and forever explaining things to them.
I had to choose another profession and learned to pilot airplanes.
I have flown a little over all parts of the world; and it is true that geography has been very useful to me.
It was in this way that I made the acquaintance of the little prince.
I lived alone, with no one really to talk to, until I had an accident with my plane in the Desert of Sahara, six years ago.
Something was broken in my engine, and as I had with me neither a mechanic nor any passengers, I set myself to attempt the difficult repairs all alone.
It was a matter of life or death for me: I had scarcely enough drinking water to last a week.
The first night I went to sleep on the sand, a thousand miles from any human habitation.
I was more isolated than a shipwrecked sailor on a raft in the middle of the ocean.
Thus you can imagine my amazement, at sunrise, when I was awakened by an odd little voice.
It said: If you please, draw me a sheep.
When a mystery is too overpowering, one dare not disobey.
I took out of my pocket a sheet of paper and my fountain pen.
But then I remembered how my studies had been concentrated on geography, history, arithmetic and grammar.
So I put my pen back and explained to the little prince that I did not know how to draw.
He answered me: That does not matter. Draw me a sheep.
But I had never drawn a sheep, so I drew for him one of the two pictures I had drawn so often.
The little prince looked at my drawing carefully, then he said: No. This sheep is already very sickly. Make me another.
So I made another drawing. My friend smiled gently and indulgently.
You see yourself that this is not a sheep. This is a ram. It has horns.
So then I did my drawing over once more. But it was rejected, too, just like the others.
This one is too old. I want a sheep that will live a long time.
I was growing impatient with this exercise of my patience, since I was in a hurry to start taking apart my engine.
I scribbled this drawing hastily and explained to my little friend: This is only his box. The sheep you asked for is inside.
I was very surprised to see a light break over the face of my young judge.
If it is really a sheep I want, he said. Is it not a very fat sheep? No, the sheep you want is not a fat one.
It is quite a small sheep. You will see that the sheep has gone to sleep. And he went off carrying his treasure.
I had still not found out about his planet, nor anything else about him, only that he came from very far away.
The little prince asked me a great many questions, but he never seemed to hear the ones I asked him.
It was from words dropped by chance that, little by little, everything was revealed to me."""

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="nlp-header">
  <h1>🧠 NLP 语义表示方法交互系统</h1>
  <p>文本分布式表示方法的交互式探索 · 从 LSA 到 FastText</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 模块一 · TF-IDF 与 LSA",
    "🔤 模块二 · Word2Vec",
    "✨ 模块三 · GloVe 词类比",
    "🚀 模块四 · FastText & Sent2Vec",
])

# ══════════════════════════════════════════════
# MODULE 1 — TF-IDF & LSA
# ══════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="info-box">
    <b>📖 学习提示</b> — TF-IDF（词频-逆文档频率）将每个文档表示为稀疏加权向量，词语权重综合考量了
    其在文档中的出现频率与在整个语料中的稀有程度。LSA（潜在语义分析）对 TF-IDF 矩阵做截断 SVD 分解，
    发掘语料的潜在语义维度，将高维稀疏词汇空间压缩为低维稠密表示。
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("#### 📝 语料输入")
        corpus_text = st.text_area(
            "输入英文文本（将按句号自动切分为文档集合）：",
            value=DEFAULT_CORPUS,
            height=280,
            key="corpus1",
        )

        vec_method = st.radio(
            "LSA 输入矩阵类型：",
            ["TF-IDF", "One-hot（CountVectorizer）"],
            horizontal=True,
        )
        n_components = st.slider("LSA 降维维数（设为 2 可直接 2D 可视化）：", 2, 20, 2, key="lsa_dim")
        run1 = st.button("▶ 开始分析", key="run1", use_container_width=True)

    with col_right:
        if run1 and corpus_text.strip():
            import nltk
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            from sklearn.decomposition import TruncatedSVD

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            # sentence split
            sentences = [s.strip() for s in corpus_text.replace("\n", " ").split(".") if len(s.strip()) > 10]

            st.markdown(f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-label">文档数量</div>
                <div class="metric-value">{len(sentences)}</div>
                <div class="metric-sub">句子（文档）</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Vectorize
            if vec_method == "TF-IDF":
                vec = TfidfVectorizer(stop_words="english", max_features=200, min_df=1)
            else:
                vec = CountVectorizer(stop_words="english", max_features=200, min_df=1)

            X = vec.fit_transform(sentences)
            vocab = vec.get_feature_names_out()

            # Top keywords
            if vec_method == "TF-IDF":
                mean_scores = np.asarray(X.mean(axis=0)).flatten()
                top_idx = mean_scores.argsort()[-10:][::-1]
                st.markdown('<div class="card"><div class="card-title">🔑 TF-IDF 权重最高的关键词 Top 5</div>', unsafe_allow_html=True)
                for rank, i in enumerate(top_idx[:5]):
                    pct = int(mean_scores[i] / mean_scores[top_idx[0]] * 100)
                    st.markdown(f"""
                    <div class="result-row">
                      <span class="result-word">#{rank+1} {vocab[i]}</span>
                      <div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>
                      <span class="result-score">{mean_scores[i]:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # LSA
            real_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
            svd = TruncatedSVD(n_components=real_components, random_state=42)
            X_doc = svd.fit_transform(X)

            word_vecs = svd.components_.T

            # Variance explained
            exp_var = svd.explained_variance_ratio_
            st.markdown(f"""
            <div class="info-box">
            <b>LSA 方差解释率</b> — 第 1 主成分：<b>{exp_var[0]:.1%}</b>
            {f"&nbsp;|&nbsp; 第 2 主成分：<b>{exp_var[1]:.1%}</b>" if len(exp_var) > 1 else ""}
            &nbsp;|&nbsp; 合计：<b>{exp_var.sum():.1%}</b>
            </div>""", unsafe_allow_html=True)

            if real_components >= 2:
                top_n = min(60, len(vocab))
                if vec_method == "TF-IDF":
                    importance = np.asarray(X.mean(axis=0)).flatten()
                    sel_idx = importance.argsort()[-top_n:][::-1]
                else:
                    sel_idx = np.arange(top_n)

                wx = word_vecs[sel_idx, 0]
                wy = word_vecs[sel_idx, 1]
                words_sel = [vocab[i] for i in sel_idx]

                fig = px.scatter(
                    x=wx, y=wy, text=words_sel,
                    labels={"x": "LSA 第 1 维", "y": "LSA 第 2 维"},
                    title="LSA 2D 词汇语义空间",
                    template="plotly_dark",
                )
                fig.update_traces(
                    textposition="top center",
                    marker=dict(size=7, color="#6366f1", opacity=0.85),
                    textfont=dict(size=10, color="#a78bfa"),
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(24,24,27,0.8)",
                    font_family="Inter",
                    title_font_size=14,
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)
        elif run1:
            st.markdown('<div class="error-box">⚠️ 请先输入语料文本。</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 3rem; color: var(--muted-foreground);">
              <div style="font-size:2.5rem; margin-bottom:0.75rem;">📊</div>
              <div style="font-size:0.9rem;">点击 <b>开始分析</b> 即可可视化 TF-IDF 关键词权重与 LSA 2D 词汇空间</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODULE 2 — Word2Vec
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="info-box">
    <b>📖 学习提示</b> — Word2Vec 通过神经网络学习稠密词向量。<b>Skip-Gram</b> 以当前词预测上下文词；
    <b>CBOW</b>（连续词袋）以上下文词预测当前词。两种架构均能使语义相似的词在向量空间中距离更近，
    相似度可通过余弦相似度（Cosine Similarity）度量。
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1.3], gap="large")

    with col_a:
        st.markdown("#### ⚙️ 训练参数")
        corpus2 = st.text_area("训练语料：", value=DEFAULT_CORPUS, height=200, key="corpus2")

        arch = st.radio("模型架构：", ["CBOW (sg=0)", "Skip-Gram (sg=1)"], horizontal=True)
        sg_val = 0 if "CBOW" in arch else 1

        window_sz = st.slider("上下文窗口大小：", 2, 10, 5, key="w2v_window")
        vec_sz = st.slider("词向量维度：", 10, 100, 50, step=10, key="w2v_dim")

        train_btn = st.button("🚀 训练 Word2Vec", key="train_w2v", use_container_width=True)

        st.markdown('<div class="section-divider"><div class="line"></div><div class="label">相似词查询</div><div class="line"></div></div>', unsafe_allow_html=True)
        query_word = st.text_input("查询与该词最相似的词：", placeholder="例如：sheep", key="w2v_query")
        query_btn = st.button("🔍 查找相似词", key="query_w2v", use_container_width=True)

    with col_b:
        if "w2v_model" not in st.session_state:
            st.session_state.w2v_model = None
            st.session_state.w2v_arch = ""

        if train_btn and corpus2.strip():
            import nltk
            from gensim.models import Word2Vec

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            sentences_raw = [s.strip() for s in corpus2.replace("\n", " ").split(".") if len(s.strip()) > 5]
            tokenized = [nltk.word_tokenize(s.lower()) for s in sentences_raw]
            tokenized = [[w for w in t if w.isalpha()] for t in tokenized]

            with st.spinner("正在训练 Word2Vec..."):
                model = Word2Vec(
                    sentences=tokenized,
                    vector_size=vec_sz,
                    window=window_sz,
                    sg=sg_val,
                    min_count=1,
                    epochs=50,
                    seed=42,
                )
            st.session_state.w2v_model = model
            st.session_state.w2v_arch = "Skip-Gram" if sg_val else "CBOW"
            st.session_state.w2v_tokenized = tokenized

            vocab_size = len(model.wv)
            st.markdown(f"""
            <div class="success-box">✅ 模型训练完成！架构：<b>{st.session_state.w2v_arch}</b> &nbsp;|&nbsp;
            词表大小：<b>{vocab_size}</b> 个词 &nbsp;|&nbsp; 向量维度：<b>{vec_sz}</b>
            </div>""", unsafe_allow_html=True)

            # 2D PCA visualization
            from sklearn.decomposition import PCA
            words_all = list(model.wv.index_to_key)[:60]
            vecs = np.array([model.wv[w] for w in words_all])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(vecs)

            fig2 = px.scatter(
                x=coords[:, 0], y=coords[:, 1], text=words_all,
                title=f"Word2Vec（{st.session_state.w2v_arch}）— PCA 2D 词向量投影",
                labels={"x": "主成分 1", "y": "主成分 2"},
                template="plotly_dark",
            )
            fig2.update_traces(
                textposition="top center",
                marker=dict(size=7, color="#a78bfa", opacity=0.85),
                textfont=dict(size=9, color="#c4b5fd"),
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(24,24,27,0.8)",
                font_family="Inter", title_font_size=13,
                margin=dict(l=20, r=20, t=40, b=20), height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

        if (query_btn or query_word) and st.session_state.w2v_model and query_word.strip():
            model = st.session_state.w2v_model
            word = query_word.strip().lower()
            try:
                similar = model.wv.most_similar(word, topn=5)
                st.markdown(f'<div class="card"><div class="card-title">🔍 与「{word}」最相似的词 — {st.session_state.w2v_arch}</div>', unsafe_allow_html=True)
                for rank, (w, score) in enumerate(similar):
                    pct = int(score * 100)
                    st.markdown(f"""
                    <div class="result-row">
                      <span class="result-word">#{rank+1} {w}</span>
                      <div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>
                      <span class="result-score">{score:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except KeyError:
                st.markdown(f'<div class="error-box">❌ 词「<b>{word}</b>」不在词表中（OOV 未登录词）。请先训练模型或换一个词。</div>', unsafe_allow_html=True)

        elif not st.session_state.w2v_model and not train_btn:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem; color: var(--muted-foreground);">
              <div style="font-size:2.5rem; margin-bottom:0.75rem;">🔤</div>
              <div>设置好参数后，点击 <b>训练 Word2Vec</b> 开始</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODULE 3 — GloVe & Analogy
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="info-box">
    <b>📖 学习提示</b> — GloVe（全局词向量）基于全局词-词共现统计矩阵训练词向量，融合了全局统计信息与局部上下文信息。
    其核心特性：<code>vec(king) − vec(man) + vec(woman) ≈ vec(queen)</code>，即词向量空间中存在语义线性关系。
    本模块使用轻量级预训练模型 <b>glove-twitter-25</b>（25 维，基于 20 亿条推文训练）。
    </div>
    """, unsafe_allow_html=True)

    @st.cache_resource(show_spinner="正在加载 GloVe 预训练模型（首次约需 30~60 秒）...")
    def load_glove():
        import os, json, gzip, shutil, hashlib
        import urllib.request
        import gensim.downloader as api
        from gensim.models import KeyedVectors

        BASE_DIR  = os.path.expanduser("~/gensim-data")
        INFO_PATH = os.path.join(BASE_DIR, "information.json")
        MODEL_DIR = os.path.join(BASE_DIR, "glove-twitter-25")
        MODEL_GZ  = os.path.join(MODEL_DIR, "glove-twitter-25.gz")
        INIT_PY   = os.path.join(MODEL_DIR, "__init__.py")
        EXPECTED_MD5 = "b6172d653b041aa0a636526a72f07de1"
        EXPECTED_SIZE = 109885004  # bytes
        GZ_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/glove-twitter-25.gz"

        # ── 工具函数：校验 gz 文件完整性 ──────────────────────────────────
        def gz_is_valid(path):
            """返回 True 表示文件存在、大小合理、可完整解压读取"""
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) < EXPECTED_SIZE * 0.95:   # 允许 5% 误差
                return False
            try:
                with gzip.open(path, "rb") as f:
                    while f.read(1024 * 1024):                  # 逐块读完整个文件
                        pass
                return True
            except Exception:
                return False

        def md5_ok(path):
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest() == EXPECTED_MD5

        # ── Step 1: 修复 information.json 结构 ────────────────────────────
        # Streamlit Cloud may preserve a broken gensim reader cache
        # ("glove-twitter-25" without load_data). Loading the raw vectors
        # directly avoids that reader module entirely.
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not gz_is_valid(MODEL_GZ):
            if os.path.exists(MODEL_GZ):
                os.remove(MODEL_GZ)
            urllib.request.urlretrieve(GZ_URL, MODEL_GZ)
        if gz_is_valid(MODEL_GZ):
            return KeyedVectors.load_word2vec_format(
                MODEL_GZ,
                binary=False,
                no_header=False,
            )

        os.makedirs(BASE_DIR, exist_ok=True)
        correct_info = {
            "corpora": {},
            "models": {
                "glove-twitter-25": {
                    "description": "Pre-trained vectors, 2B tweets, 27B tokens, 1.2M vocab, uncased.",
                    "parameters": {"dimensions": 25},
                    "file_size": EXPECTED_SIZE,
                    "base_dataset": "Twitter",
                    "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/__init__.py",
                    "license": "http://opendatacommons.org/licenses/pddl/",
                    "parts": 1,
                    "checksum": EXPECTED_MD5,
                    "file_name": "glove-twitter-25.gz",
                }
            },
        }
        needs_fix = True
        if os.path.exists(INFO_PATH):
            try:
                with open(INFO_PATH) as f:
                    d = json.load(f)
                if "corpora" in d and "models" in d:
                    needs_fix = False
            except Exception:
                pass
        if needs_fix:
            with open(INFO_PATH, "w") as f:
                json.dump(correct_info, f)

        # ── Step 2: 检测并清理损坏的 gz 文件（EOFError 根源）─────────────
        if os.path.exists(MODEL_GZ) and not gz_is_valid(MODEL_GZ):
            os.remove(MODEL_GZ)   # 删除不完整 / 损坏的文件，强制重新下载

        # ── Step 3: 文件完整则直接加载 ────────────────────────────────────
        if os.path.exists(MODEL_GZ) and os.path.exists(INIT_PY):
            try:
                return api.load("glove-twitter-25")
            except Exception:
                pass
            try:
                return KeyedVectors.load_word2vec_format(MODEL_GZ, binary=False, no_header=False)
            except Exception:
                pass

        # ── Step 4: 下载（gensim downloader 优先，失败则手动 urllib）──────
        os.makedirs(MODEL_DIR, exist_ok=True)
        init_url = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/__init__.py"
        gz_url   = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/glove-twitter-25.gz"

        try:
            return api.load("glove-twitter-25")
        except Exception:
            pass

        # 手动下载
        try:
            if not os.path.exists(INIT_PY):
                urllib.request.urlretrieve(init_url, INIT_PY)
            if not os.path.exists(MODEL_GZ):
                urllib.request.urlretrieve(gz_url, MODEL_GZ)
        except Exception as e:
            raise RuntimeError(
                f"网络下载失败，请在终端手动执行以下命令后重启 Streamlit：\n"
                f"python -c \"import gensim.downloader as api; api.load('glove-twitter-25')\"\n\n"
                f"原始错误：{e}"
            )

        # 下载后再次校验
        if not gz_is_valid(MODEL_GZ):
            if os.path.exists(MODEL_GZ):
                os.remove(MODEL_GZ)
            raise RuntimeError(
                "下载的 GloVe 文件不完整或已损坏，请删除以下目录后重试：\n"
                f"{MODEL_DIR}\n\n"
                "或在终端手动运行：\n"
                "python -c \"import gensim.downloader as api; api.load('glove-twitter-25')\""
            )

        return api.load("glove-twitter-25")

    col_g1, col_g2 = st.columns([1, 1], gap="large")

    with col_g1:
        st.markdown("#### 🔄 词类比计算器")
        st.markdown("""
        <div class="card" style="margin-bottom:1rem;">
          <div class="card-title">公式：Result = A − B + C</div>
          <div style="font-size:0.82rem; color:var(--muted-foreground); margin-bottom:0.5rem;">
          示例：A=<i>king</i>，B=<i>man</i>，C=<i>woman</i> → <i>queen</i>
          </div>
        </div>
        """, unsafe_allow_html=True)

        word_a = st.text_input("词 A：", value="king", key="ga")
        word_b = st.text_input("词 B：", value="man", key="gb")
        word_c = st.text_input("词 C：", value="woman", key="gc")
        analogy_btn = st.button("🔮 计算词类比", key="analogy_btn", use_container_width=True)

        st.markdown('<div class="section-divider"><div class="line"></div><div class="label">词义相似度</div><div class="line"></div></div>', unsafe_allow_html=True)
        st.markdown("#### 📐 词义相似度计算")
        sim_w1 = st.text_input("词语 1：", value="prince", key="sim1")
        sim_w2 = st.text_input("词语 2：", value="king", key="sim2")
        sim_btn = st.button("📊 计算相似度", key="sim_btn", use_container_width=True)

    with col_g2:
        glove_loaded = False
        try:
            glove = load_glove()
            glove_loaded = True
            st.markdown('<div class="success-box">✅ GloVe（glove-twitter-25）预训练模型加载成功</div>', unsafe_allow_html=True)
        except Exception as e:
            err_msg = str(e)
            st.markdown(f"""
            <div class="error-box">
            ❌ <b>GloVe 加载失败</b><br><br>
            {err_msg}<br><br>
            <b>解决方法：</b>请在终端执行以下命令预先下载模型，然后重启 Streamlit：<br>
            <code>python -c "import gensim.downloader as api; api.load('glove-twitter-25')"</code>
            </div>
            """, unsafe_allow_html=True)

        if glove_loaded and analogy_btn:
            a, b, c = word_a.strip().lower(), word_b.strip().lower(), word_c.strip().lower()
            try:
                results = glove.most_similar(positive=[a, c], negative=[b], topn=5)
                formula_html = f"<code>vec(<b>{a}</b>) − vec(<b>{b}</b>) + vec(<b>{c}</b>)</code>"
                st.markdown(f'<div class="card"><div class="card-title">🔮 词类比结果 — {formula_html}</div>', unsafe_allow_html=True)
                for rank, (w, score) in enumerate(results):
                    pct = int(score * 100)
                    badge = '<span class="badge badge-primary">最佳匹配</span>' if rank == 0 else ""
                    st.markdown(f"""
                    <div class="result-row">
                      <span class="result-word">#{rank+1} {w} {badge}</span>
                      <div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>
                      <span class="result-score">{score:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except KeyError as e:
                st.markdown(f'<div class="error-box">❌ 该词不在 GloVe 词表中：{e}</div>', unsafe_allow_html=True)

        if glove_loaded and sim_btn:
            w1, w2 = sim_w1.strip().lower(), sim_w2.strip().lower()
            try:
                score = glove.similarity(w1, w2)
                if score > 0.7: badge_cls = "badge-success"
                elif score > 0.4: badge_cls = "badge-warning"
                else: badge_cls = "badge-danger"
                label = "高度相似" if score > 0.7 else ("中度相似" if score > 0.4 else "低相似")
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">📐 余弦相似度</div>
                  <div style="display:flex; align-items:center; gap:1rem; margin-top:0.5rem;">
                    <span style="font-size:2rem; font-weight:700; color:var(--foreground);">{score:.4f}</span>
                    <span class="badge {badge_cls}">{label}</span>
                  </div>
                  <div style="font-size:0.82rem; color:var(--muted-foreground); margin-top:0.5rem;">
                  sim("<b>{w1}</b>", "<b>{w2}</b>")
                  </div>
                </div>
                """, unsafe_allow_html=True)

                sim_a = glove.most_similar(w1, topn=5)
                sim_b = glove.most_similar(w2, topn=5)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="card-title">「{w1}」的近邻词</div>', unsafe_allow_html=True)
                    for w, s in sim_a:
                        st.markdown(f'<span class="badge badge-info">{w}</span>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="card-title">「{w2}」的近邻词</div>', unsafe_allow_html=True)
                    for w, s in sim_b:
                        st.markdown(f'<span class="badge badge-primary">{w}</span>', unsafe_allow_html=True)

            except KeyError as e:
                st.markdown(f'<div class="error-box">❌ 词不在词表中：{e}</div>', unsafe_allow_html=True)

        if glove_loaded and not analogy_btn and not sim_btn:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem; color:var(--muted-foreground);">
              <div style="font-size:2.5rem; margin-bottom:0.75rem;">✨</div>
              <div>输入词语，点击上方按钮即可探索 GloVe 词向量空间</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MODULE 4 — FastText & Sent2Vec
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="info-box">
    <b>📖 学习提示</b> — FastText 在 Word2Vec 基础上引入<b>子词（字符 n-gram）</b>信息，
    对每个词的向量由其所有子词向量叠加而成，从而能够处理<b>词表外（OOV）</b>词语。
    <b>Sent2Vec</b>（简化实现）将句子中所有词向量做<b>均值池化（Average Pooling）</b>，
    得到固定维度的全局句子向量。
    </div>
    """, unsafe_allow_html=True)

    corpus4 = st.text_area("FastText 训练语料：", value=DEFAULT_CORPUS, height=160, key="corpus4")
    train_ft_btn = st.button("🚀 训练 FastText 模型", key="train_ft", use_container_width=False)

    if "ft_model" not in st.session_state:
        st.session_state.ft_model = None
    if "ft_tokenized" not in st.session_state:
        st.session_state.ft_tokenized = None

    if train_ft_btn and corpus4.strip():
        import nltk
        from gensim.models import FastText

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        sentences_raw = [s.strip() for s in corpus4.replace("\n", " ").split(".") if len(s.strip()) > 5]
        tokenized = [nltk.word_tokenize(s.lower()) for s in sentences_raw]
        tokenized = [[w for w in t if w.isalpha()] for t in tokenized]

        with st.spinner("正在训练 FastText..."):
            ft = FastText(sentences=tokenized, vector_size=50, window=5, min_count=1, epochs=50, seed=42)
        st.session_state.ft_model = ft
        st.session_state.ft_tokenized = tokenized

        # Also keep a w2v model for OOV comparison
        from gensim.models import Word2Vec
        with st.spinner("正在训练 Word2Vec（用于 OOV 对比）..."):
            w2v_cmp = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, epochs=50, seed=42)
        st.session_state.w2v_cmp = w2v_cmp

        st.markdown(f'<div class="success-box">✅ FastText 训练完成！词表大小：<b>{len(ft.wv)}</b> 个词 · 可通过子词 n-gram 处理任意 OOV 词</div>', unsafe_allow_html=True)

    # ── OOV TEST ──
    st.markdown('<div class="section-divider"><div class="line"></div><div class="label">OOV 测试 — 子词鲁棒性对比</div><div class="line"></div></div>', unsafe_allow_html=True)
    col_oov1, col_oov2 = st.columns([1, 1.2], gap="large")

    with col_oov1:
        st.markdown("#### 🔬 OOV 未登录词测试")
        st.markdown("""
        <div class="card" style="margin-bottom:0.75rem;">
          <div style="font-size:0.82rem; color:var(--muted-foreground);">
          输入一个拼写错误或自造词语。Word2Vec 将因找不到该词而抛出 KeyError（OOV），
          而 FastText 会利用字符级子词 n-gram 仍然计算出有意义的向量表示。
          </div>
        </div>
        """, unsafe_allow_html=True)
        oov_word = st.text_input("测试词（可输入拼写错误的词）：", value="princce", key="oov_input")
        oov_btn = st.button("🧪 测试 OOV 处理能力", key="oov_btn", use_container_width=True)

    with col_oov2:
        if oov_btn and st.session_state.ft_model:
            ft = st.session_state.ft_model
            w2v_cmp = st.session_state.w2v_cmp

            # Word2Vec attempt
            st.markdown("**Word2Vec 结果：**")
            try:
                w2v_sim = w2v_cmp.wv.most_similar(oov_word.lower(), topn=3)
                for w, s in w2v_sim:
                    st.markdown(f'<span class="badge badge-success">{w}: {s:.3f}</span>', unsafe_allow_html=True)
            except KeyError:
                st.markdown(f'<div class="error-box">❌ <b>Word2Vec</b>：KeyError — 「<b>{oov_word}</b>」是未登录词（OOV），无法获取词向量。</div>', unsafe_allow_html=True)

            # FastText attempt
            st.markdown("**FastText 结果：**")
            try:
                ft_sim = ft.wv.most_similar(oov_word.lower(), topn=5)
                st.markdown(f'<div class="success-box">✅ <b>FastText</b> 通过子词 n-gram 成功处理 OOV 词：</div>', unsafe_allow_html=True)
                st.markdown('<div class="card"><div class="card-title">FastText 最相似词</div>', unsafe_allow_html=True)
                for rank, (w, score) in enumerate(ft_sim):
                    pct = int(score * 100)
                    st.markdown(f"""
                    <div class="result-row">
                      <span class="result-word">#{rank+1} {w}</span>
                      <div class="score-bar-bg"><div class="score-bar" style="width:{pct}%"></div></div>
                      <span class="result-score">{score:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">FastText 错误：{e}</div>', unsafe_allow_html=True)

        elif oov_btn and not st.session_state.ft_model:
            st.markdown('<div class="warning-box">⚠️ 请先训练 FastText 模型。</div>', unsafe_allow_html=True)

    # ── SENT2VEC ──
    st.markdown('<div class="section-divider"><div class="line"></div><div class="label">Sent2Vec — 句向量与语义相似度（均值池化）</div><div class="line"></div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Sent2Vec 原理</b>：将句子中每个词的词向量做<b>逐元素均值（Average Pooling）</b>，
    得到固定维度的全局句子向量。对两个句子向量计算余弦相似度，即可衡量句子层面的语义接近程度。
    <br><br>
    <code>sent_vec(S) = (1/N) Σ vec(wᵢ) &nbsp;&nbsp; 其中 wᵢ ∈ S</code>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1, 1], gap="large")
    with col_s1:
        sent1 = st.text_area("句子 A：", value="The little prince came from a small planet.", height=100, key="sent1")
    with col_s2:
        sent2 = st.text_area("句子 B：", value="A tiny boy lived on a distant star.", height=100, key="sent2")

    sent_btn = st.button("📐 计算句子相似度", key="sent_btn", use_container_width=False)

    if sent_btn:
        if not st.session_state.ft_model:
            st.markdown('<div class="warning-box">⚠️ 请先训练 FastText 模型。</div>', unsafe_allow_html=True)
        else:
            import nltk
            ft = st.session_state.ft_model

            def sentence_vector(sentence, model):
                try:
                    tokens = nltk.word_tokenize(sentence.lower())
                except:
                    tokens = sentence.lower().split()
                tokens = [w for w in tokens if w.isalpha()]
                vecs = [model.wv[w] for w in tokens if w in model.wv or True]
                if not vecs:
                    return None, []
                return np.mean(vecs, axis=0), tokens

            from numpy.linalg import norm
            vec1, tok1 = sentence_vector(sent1, ft)
            vec2, tok2 = sentence_vector(sent2, ft)

            if vec1 is not None and vec2 is not None:
                cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-9)

                if cos_sim > 0.85: level, cls = "极高相似", "badge-success"
                elif cos_sim > 0.65: level, cls = "高度相似", "badge-success"
                elif cos_sim > 0.45: level, cls = "中度相似", "badge-warning"
                else: level, cls = "低相似", "badge-danger"

                c_r1, c_r2, c_r3 = st.columns(3)
                with c_r1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">余弦相似度</div>
                      <div class="metric-value">{cos_sim:.4f}</div>
                      <div class="metric-sub"><span class="badge {cls}">{level}</span></div>
                    </div>""", unsafe_allow_html=True)
                with c_r2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">句子 A 词数</div>
                      <div class="metric-value">{len(tok1)}</div>
                      <div class="metric-sub">参与向量化的词语</div>
                    </div>""", unsafe_allow_html=True)
                with c_r3:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">句子 B 词数</div>
                      <div class="metric-value">{len(tok2)}</div>
                      <div class="metric-sub">参与向量化的词语</div>
                    </div>""", unsafe_allow_html=True)

                dims = min(20, len(vec1))
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    name="句子 A", x=[f"d{i}" for i in range(dims)],
                    y=vec1[:dims], marker_color="#6366f1", opacity=0.8,
                ))
                fig_s.add_trace(go.Bar(
                    name="句子 B", x=[f"d{i}" for i in range(dims)],
                    y=vec2[:dims], marker_color="#a78bfa", opacity=0.8,
                ))
                fig_s.update_layout(
                    barmode="group", title=f"句向量维度对比（前 {dims} 维）· 余弦相似度 = {cos_sim:.4f}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(24,24,27,0.8)",
                    font_family="Inter", template="plotly_dark",
                    legend=dict(orientation="h", y=1.12),
                    height=320, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_s, use_container_width=True)
