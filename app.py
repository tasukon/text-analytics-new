import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from janome.tokenizer import Tokenizer
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
import itertools
import re
import time

# --- 1. アプリの設定 ---
st.set_page_config(page_title="Text Analytics V6", layout="wide")

# セッションステート初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'user_stopwords' not in st.session_state:
    st.session_state.user_stopwords = []
if 'step' not in st.session_state:
    st.session_state.step = 1

# 基本のストップワード
DEFAULT_STOPWORDS = [
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や",
    "れる", "など", "ない", "この", "ため", "その", "よう", "また", "もの",
    "ます", "です", "さん", "ちゃん", "くん", "あっ", "あり", "いっ", "う",
    "か", "せる", "たい", "だけ", "たち", "ついて", "でき", "なり", "の",
    "ばかり", "ほど", "まで", "まま", "よう", "より", "わたし", "それ", "これ",
    "回答", "なし", "特になし", "特に", "ため", "てき", "それら"
]

# --- 2. 関数定義 ---

def classify_columns(df):
    """属性(フィルタ用)とテキスト(分析用)を自動判定"""
    filter_cols = [] 
    text_cols = []   

    for col in df.columns:
        unique_count = df[col].nunique()
        # 50種類未満なら「属性」とみなす
        if unique_count < 50:
            filter_cols.append(col)
        # それ以外で文字型なら「テキスト」とみなす
        elif df[col].dtype == 'object':
            text_cols.append(col)
            
    return filter_cols, text_cols

@st.cache_data
def get_tokens(text, stop_words):
    """形態素解析"""
    t = Tokenizer()
    tokens = []
    if not isinstance(text, str):
        return []
    
    japanese_pattern = re.compile(r'[ぁ-んァ-ン一-龥]')
    
    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if (pos in ['名詞', '動詞', '形容詞'] and 
            len(base) > 1 and 
            base not in stop_words and 
            japanese_pattern.search(base)):
            tokens.append(base)
    return tokens

@st.cache_data
def create_network(tokens_list, top_n, min_edge):
    """共起ネットワーク生成"""
    pair_list = []
    for tokens in tokens_list:
        if len(tokens) >= 2:
            pair_list.extend(itertools.combinations(tokens, 2))
    
    c = Counter(pair_list)
    top_pairs = c.most_common(top_n)
    
    G = nx.Graph()
    for (u, v), weight in top_pairs:
        if weight >= min_edge:
            G.add_edge(u, v, weight=weight)
    return G

# --- 3. メイン処理 ---

# === STEP 1: データ読み込み ===
if st.session_state.step == 1:
    st.title("📂 Step 1: データの読み込み")
    st.info("分析したい CSV または Excel ファイルをアップロードしてください。")
    
    uploaded_file = st.file_uploader("ファイルをドラッグ＆ドロップ", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.step = 2
            st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")

# === STEP 2: 除外ワード設定 (全テキスト一括) ===
elif st.session_state.step == 2:
    
    # --- レイアウト: タイトルと「次へ」ボタンを横並びに ---
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("🧹 Step 2: データクリーニング")
        st.markdown("ファイル内の**すべての文章**から、頻出単語を集計しました。不要な言葉を除外してください。")
    
    df = st.session_state.df
    filter_candidates, text_candidates = classify_columns(df)
    
    # 完了ボタン（上部に配置）
    with header_col2:
        st.write("") # 余白調整
        if st.button("設定完了！分析画面へ (Step 3) >>", type="primary", use_container_width=True):
            st.session_state.text_candidates = text_candidates # 自動判定したテキスト列
            st.session_state.filter_candidates = filter_candidates # 自動判定した属性列
            st.session_state.step = 3
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("全体ランキング (TOP30)")
        
        # 全テキスト列を結合して分析
        current_stop = DEFAULT_STOPWORDS + st.session_state.user_stopwords
        
        full_text_data = ""
        # 自動判定された「テキスト列」の中身を全部つなげる
        target_cols = text_candidates if text_candidates else df.columns
        
        for col in target_cols:
            full_text_data += " " + " ".join(df[col].dropna().astype(str).tolist())
        
        tokens = get_tokens(full_text_data, current_stop)
        
        if tokens:
            c = Counter(tokens)
            words, counts = zip(*c.most_common(30))
            
            # グラフ描画
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.barh(words, counts, color='gray')
            ax.invert_yaxis()
            ax.set_title("すべての文章の合計")
            st.pyplot(fig)
        else:
            st.warning("テキストデータが見つかりません。")

    with col2:
        st.subheader("除外ワードの追加")
        st.info("左のグラフを見て、分析に不要な単語を入力してください。")
        
        new_word = st.text_input("除外したい単語 (入力してEnter)", placeholder="例: 私 思う アンケート")
        if new_word:
            words = new_word.split()
            added = []
            for w in words:
                if w not in st.session_state.user_stopwords:
                    st.session_state.user_stopwords.append(w)
                    added.append(w)
            if added:
                st.success(f"除外しました: {added}")
                time.sleep(0.5)
                st.rerun()
        
        st.write("🚫 **現在の除外リスト:**")
        st.write(st.session_state.user_stopwords)
        
        if st.button("リセット"):
            st.session_state.user_stopwords = []
            st.rerun()

# === STEP 3: 最終分析 (多重フィルタリング & 可視化) ===
elif st.session_state.step == 3:
    st.title("📊 Step 3: 詳細分析")
    
    df = st.session_state.df
    text_candidates = st.session_state.text_candidates
    filter_candidates = st.session_state.filter_candidates
    stop_words = DEFAULT_STOPWORDS + st.session_state.user_stopwords

    # --- サイドバー: フィルタリング設定 ---
    st.sidebar.header("🔍 データの絞り込み")
    
    # フィルタリング処理
    df_filtered = df.copy()
    
    # 自動判定された属性列ごとに、選択ボックスを作る
    active_filters = []
    for col in filter_candidates:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(f"{col}", unique_vals)
        
        if selected:
            df_filtered = df_filtered[df_filtered[col].isin(selected)]
            
    st.sidebar.markdown("---")
    st.sidebar.write(f"**分析対象:** {len(df_filtered)} 行 / {len(df)} 行")
    
    # 戻るボタン
    if st.sidebar.button("Step 2 (除外設定) に戻る"):
        st.session_state.step = 2
        st.rerun()
    if st.sidebar.button("Step 1 (ファイル選択) に戻る"):
        st.session_state.df = None
        st.session_state.step = 1
        st.rerun()

    # --- メインエリア: 可視化 ---
    
    if len(df_filtered) == 0:
        st.error("条件に合うデータが0件です。フィルタ条件を緩めてください。")
    else:
        # Step 3では、対象となる全テキスト列を結合して表示
        full_text = ""
        target_cols = text_candidates if text_candidates else df.columns
        for col in target_cols:
            full_text += " " + " ".join(df_filtered[col].dropna().astype(str).tolist())
            
        tokens = get_tokens(full_text, stop_words)

        if not tokens:
            st.warning("表示できる単語がありません。")
        else:
            tab1, tab2, tab3 = st.tabs(["☁️ ワードクラウド", "🕸️ 共起ネットワーク", "📈 ランキング"])
            
            with tab1:
                st.subheader("ワードクラウド")
                try:
                    wc = WordCloud(
                        background_color="white", width=800, height=500,
                        regexp=r"[\w']+", font_path="IPAexGothic.ttf"
                    ).generate(" ".join(tokens))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except:
                    st.error("フォント読込エラー")

            with tab2:
                st.subheader("共起ネットワーク")
                col1, col2 = st.columns(2)
                with col1:
                    net_top = st.slider("エッジ数", 10, 200, 50, key='net1')
                with col2:
                    min_edge = st.slider("最小共起回数", 1, 10, 2, key='net2')
                
                # 行ごとのリスト作成（全テキスト列を結合）
                sentences = []
                for i, row in df_filtered.iterrows():
                    row_text = " ".join([str(row[c]) for c in target_cols if pd.notna(row[c])])
                    sentences.append(row_text)

                tokens_list = [get_tokens(s, stop_words) for s in sentences]
                G = create_network(tokens_list, net_top, min_edge)
                
                if G.number_of_nodes() > 0:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    pos = nx.spring_layout(G, k=0.6, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#66b3ff', alpha=0.9, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
                    nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=11, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("つながりが見つかりませんでした。")

            with tab3:
                st.subheader("頻出単語ランキング")
                c = Counter(tokens)
                common = c.most_common(20)
                words, counts = zip(*common)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(words, counts, color='skyblue')
