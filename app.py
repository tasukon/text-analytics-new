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
st.set_page_config(page_title="Text Analytics V8", layout="wide")

# セッションステート初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'user_stopwords' not in st.session_state:
    st.session_state.user_stopwords = []

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
        if unique_count < 50:
            filter_cols.append(col)
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

# === ファイル読み込みエリア ===
if st.session_state.df is None:
    st.title("📂 データの読み込み")
    st.markdown("分析したい **CSV** または **Excel** ファイルをアップロードしてください。")
    uploaded_file = st.file_uploader("ファイルをドラッグ＆ドロップ", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")

# === 分析ダッシュボード ===
else:
    df = st.session_state.df
    filter_candidates, text_candidates = classify_columns(df)
    
    # --- サイドバー: 設定エリア (除外設定＆フィルタ) ---
    st.sidebar.title("⚙️ 設定パネル")
    
    # 1. 除外ワード設定 (Step 2の機能をここに統合)
    with st.sidebar.expander("🚫 除外ワードの設定", expanded=True):
        st.write(f"現在: {len(st.session_state.user_stopwords)} 語を除外中")
        new_word = st.text_input("除外したい単語を入力", placeholder="入力してEnter")
        if new_word:
            words = new_word.split()
            for w in words:
                if w not in st.session_state.user_stopwords:
                    st.session_state.user_stopwords.append(w)
            st.rerun()
        
        if st.button("除外リストをリセット"):
            st.session_state.user_stopwords = []
            st.rerun()
            
    stop_words = DEFAULT_STOPWORDS + st.session_state.user_stopwords

    # 2. フィルタリング設定
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 データの絞り込み")
    
    df_filtered = df.copy()
    for col in filter_candidates:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(f"{col}", unique_vals)
        if selected:
            df_filtered = df_filtered[df_filtered[col].isin(selected)]
            
    st.sidebar.write(f"対象: {len(df_filtered)} / {len(df)} 件")
    
    if st.sidebar.button("別のファイルを読み込む"):
        st.session_state.df = None
        st.session_state.user_stopwords = []
        st.rerun()

    # --- メインエリア: 分析結果 ---
    
    # モード切替スイッチ
    mode = st.radio("表示モード", ["全体分析", "⚔️ グループ比較"], horizontal=True)

    # ターゲット列（テキスト）の結合
    target_cols = text_candidates if text_candidates else df.columns
    
    if len(df_filtered) == 0:
        st.error("データが0件です。絞り込み条件を解除してください。")
        
    # === A. 全体分析モード ===
    elif mode == "全体分析":
        full_text = ""
        for col in target_cols:
            full_text += " " + " ".join(df_filtered[col].dropna().astype(str).tolist())
        tokens = get_tokens(full_text, stop_words)

        if not tokens:
            st.warning("表示できる単語がありません。")
        else:
            tab1, tab2, tab3 = st.tabs(["☁️ ワードクラウド", "🕸️ つながりマップ", "📈 ランキング"])
            
            with tab1:
                st.markdown("#### 全体の傾向 (直感的に見る)")
                try:
                    wc = WordCloud(
                        background_color="white", width=900, height=500,
                        regexp=r"[\w']+", font_path="IPAexGothic.ttf"
                    ).generate(" ".join(tokens))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except:
                    st.error("フォント読込エラー")

            with tab2:
                st.markdown("#### 単語のつながり (共起ネットワーク)")
                # 親切な説明 (V7の良さを継承)
                st.info("💡 **見方のヒント**: 太い線でつながっている単語は、セットで使われている言葉です。")
                
                c1, c2 = st.columns(2)
                net_top = c1.slider("表示単語数", 10, 150, 50)
                min_edge = c2.slider("最小の線の太さ", 1, 10, 2)
                
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
                    st.warning("つながりが見つかりません。")

            with tab3:
                st.markdown("#### 頻出語ランキング")
                c = Counter(tokens)
                words, counts = zip(*c.most_common(20))
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(words, counts, color='skyblue')
                ax.invert_yaxis()
                st.pyplot(fig)

    # === B. グループ比較モード ===
    elif mode == "⚔️ グループ比較":
        st.markdown("#### 2つのグループの違いを見比べる")
        
        # 比較の設定
        if not filter_candidates:
            st.error("比較できる属性列（クラスや性別など）が見つかりません。")
        else:
            col_comp_1, col_comp_2, col_comp_3 = st.columns(3)
            target_attr = col_comp_1.selectbox("どの項目で分けますか？", filter_candidates)
            
            unique_vals = sorted(df_filtered[target_attr].dropna().unique().tolist())
            if len(unique_vals) < 2:
                st.warning("比較するためのデータが足りません（1種類しかありません）。")
            else:
                val_a = col_comp_2.selectbox("グループA (左)", unique_vals, index=0)
                val_b = col_comp_3.selectbox("グループB (右)", unique_vals, index=min(1, len(unique_vals)-1))

                # データ分割 & トークン化
                df_a = df_filtered[df_filtered[target_attr] == val_a]
                df_b = df_filtered[df_filtered[target_attr] == val_b]
                
                def get_text_tokens(d):
                    txt = ""
                    for c in target_cols:
                        txt += " " + " ".join(d[c].dropna().astype(str).tolist())
                    return get_tokens(txt, stop_words)

                tokens_a = get_text_tokens(df_a)
                tokens_b = get_text_tokens(df_b)

                # 左右に並べて表示
                c_left, c_right = st.columns(2)
                
                with c_left:
                    st.info(f"🟦 {val_a} ({len(df_a)}件)")
                    if tokens_a:
                        wc_a = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_a))
                        fig_a, ax_a = plt.subplots()
                        ax_a.imshow(wc_a, interpolation='bilinear')
                        ax_a.axis("off")
                        st.pyplot(fig_a)
                    else:
                        st.write("データなし")

                with c_right:
                    st.success(f"🟧 {val_b} ({len(df_b)}件)")
                    if tokens_b:
                        wc_b = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_b))
                        fig_b, ax_b = plt.subplots()
                        ax_b.imshow(wc_b, interpolation='bilinear')
                        ax_b.axis("off")
                        st.pyplot(fig_b)
                    else:
                        st.write("データなし")
