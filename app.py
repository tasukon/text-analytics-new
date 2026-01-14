import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from janome.tokenizer import Tokenizer
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
import itertools
import re # 正規表現を使うためのライブラリ

# --- 1. アプリの基本設定 ---
st.set_page_config(
    page_title="Text Analytics App",
    page_icon="📊",
    layout="wide"
)

# セッションステート初期化
if 'df' not in st.session_state:
    st.session_state.df = None

# ストップワード（意味のない単語リスト）
DEFAULT_STOPWORDS = [
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や",
    "れる", "など", "ない", "この", "ため", "その", "よう", "また", "もの",
    "ます", "です", "さん", "ちゃん", "くん", "あっ", "あり", "いっ", "う",
    "か", "せる", "たい", "だけ", "たち", "ついて", "でき", "なり", "の",
    "ばかり", "ほど", "まで", "まま", "よう", "より", "わたし", "それ", "これ"
]

# --- 2. 関数定義 ---

@st.cache_data
def get_tokens(text, stop_words):
    """テキストから名詞・動詞・形容詞を抽出（ゴミとり機能付き）"""
    t = Tokenizer()
    tokens = []
    if not isinstance(text, str):
        return []
    
    # 日本語の文字（ひらがな、カタカナ、漢字）が含まれているかチェックする正規表現
    japanese_pattern = re.compile(r'[ぁ-んァ-ン一-龥]')

    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        
        # 条件：
        # 1. 名詞・動詞・形容詞である
        # 2. 1文字より長い（"あ" "い" などを省く）
        # 3. ストップワードに含まれていない
        # 4. 【追加】日本語の文字を含んでいる（記号や数字だけのものを省く）
        if (pos in ['名詞', '動詞', '形容詞'] and 
            len(base) > 1 and 
            base not in stop_words and 
            japanese_pattern.search(base)):
            
            tokens.append(base)
    return tokens

@st.cache_data
def create_cooccurrence_network(tokens_list, top_n=50, min_edge_weight=1):
    """共起ネットワーク作成"""
    pair_list = []
    for tokens in tokens_list:
        if len(tokens) >= 2:
            pair_list.extend(itertools.combinations(tokens, 2))
    
    c = Counter(pair_list)
    top_pairs = c.most_common(top_n)
    
    G = nx.Graph()
    for (u, v), weight in top_pairs:
        if weight >= min_edge_weight:
            G.add_edge(u, v, weight=weight)
    return G

@st.cache_data
def create_demo_data():
    """デモデータ生成"""
    data = {
        '学年': ['1年', '1年', '2年', '2年', '3年', '3年', '1年', '2年', '3年', '1年'],
        '自由記述': [
            '野性味あふれる人材になりたいし、価値創造も重要だと思う。(?)', # 記号混じりテスト
            '新しい価値を作るためには、野性的な勘が必要だと感じる！',
            '学校生活で野性味を磨き、社会で活躍したい。12345', # 数字混じりテスト
            '価値創造人材とは、失敗を恐れずに挑戦する人のことだ。',
            '勉強だけでなく、部活動でも野性味を出していきたい。',
            '将来はクリエイティブな仕事で価値を生み出したい。',
            '野性味とは、困難に立ち向かう強さのことだと思う。',
            '仲間と協力して新しい価値を創造することが目標です。',
            'もっと自由に、野性的に生きていきたい。',
            '価値創造のためには、基礎的な知識も大切だ。'
        ]
    }
    return pd.DataFrame(data)

# --- 3. 画面表示 ---

if st.session_state.df is None:
    # === Step 1: ファイルアップロード画面 ===
    st.title("📂 テキスト分析アプリ")
    st.info("下のボックスから CSV または Excelファイル を読み込んでください")
    
    uploaded_file = st.file_uploader("ファイルをここにドラッグ＆ドロップ", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            with st.spinner("読み込み中..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")

    st.markdown("---")
    if st.button("まずはデモデータで試す"):
        st.session_state.df = create_demo_data()
        st.rerun()

else:
    # === Step 2: 分析画面 ===
    df = st.session_state.df
    
    # サイドバー設定
    with st.sidebar:
        st.header("メニュー")
        # リセットボタン
        if st.button("🔄 最初に戻る"):
            st.session_state.df = None
            st.rerun()
        st.markdown("---")
        
        # 分析対象の選択
        st.subheader("分析設定")
        all_cols = df.columns
        target_col = st.selectbox("分析する列", all_cols, index=len(all_cols)-1)

    st.title(f"📊 分析結果: {target_col}")

    # タブ設定
    tab1, tab2, tab3, tab4 = st.tabs(["📂 データ", "📈 ランキング", "☁️ ワードクラウド", "🕸️ 共起ネットワーク"])

    with tab1:
        st.dataframe(df)

    with tab2:
        st.subheader("頻出単語ランキング")
        # スライダー
        top_n = st.slider("表示件数", 5, 100, 20, key='bar_slider')
        
        if st.button("グラフを表示", key='btn_bar'):
            with st.spinner("集計中..."):
                text_data = " ".join(df[target_col].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                
                if tokens:
                    counter = Counter(tokens)
                    words, counts = zip(*counter.most_common(top_n))
                    
                    # --- 【改善点】グラフの高さを自動調整 ---
                    # 1データにつき 0.4インチの高さを確保する（最低でも6インチ）
                    dynamic_height = max(6, len(words) * 0.4)
                    
                    fig, ax = plt.subplots(figsize=(10, dynamic_height))
                    ax.barh(words, counts, color='skyblue')
                    ax.invert_yaxis() # 上位を上に
                    
                    # グリッド線などを入れて見やすく
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    ax.set_title(f"出現回数ランキング (TOP {top_n})")
                    
                    st.pyplot(fig)
                else:
                    st.warning("分析できる単語が見つかりませんでした。")

    with tab3:
        st.subheader("ワードクラウド")
        if st.button("作成する", key='btn_wc'):
            with st.spinner("描画中..."):
                text_data = " ".join(df[target_col].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                text_space_sep = " ".join(tokens)
                try:
                    wc = WordCloud(
                        background_color="white", width=800, height=500,
                        regexp=r"[\w']+", font_path="IPAexGothic.ttf"
                    ).generate(text_space_sep)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.error("エラー: フォントファイルを確認してください")

    with tab4:
        st.subheader("共起ネットワーク")
        col1, col2 = st.columns(2)
        with col1:
            net_top_n = st.slider("エッジ数", 10, 200, 50, key='net_slider')
        with col2:
            min_edge = st.slider("最小共起回数", 1, 10, 1, key='net_edge')

        if st.button("ネットワークを表示", key='btn_net'):
            with st.spinner("構築中..."):
                sentences = df[target_col].dropna().astype(str).tolist()
                tokens_list = [get_tokens(s, DEFAULT_STOPWORDS) for s in sentences]
                G = create_cooccurrence_network(tokens_list, top_n=net_top_n, min_edge_weight=min_edge)
                
                if G.number_of_nodes() > 0:
                    fig, ax = plt.subplots(figsize=(12, 12))
                    pos = nx.spring_layout(G, k=0.5, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#a0cbe2', alpha=0.9, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
                    nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=11, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("つながりが見つかりませんでした。条件を緩めてください。")
