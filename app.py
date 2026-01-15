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
st.set_page_config(page_title="Text Analytics V13", layout="wide")

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
    """形態素解析 (除外ワード適用)"""
    t = Tokenizer()
    tokens = []
    if not isinstance(text, str):
        return []
    
    stop_set = set(stop_words)
    japanese_pattern = re.compile(r'[ぁ-んァ-ン一-龥]')
    
    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if (pos in ['名詞', '動詞', '形容詞'] and 
            len(base) > 1 and 
            base not in stop_set and 
            japanese_pattern.search(base)):
            tokens.append(base)
    return tokens

@st.cache_data
def create_network(tokens_list, top_n, min_edge):
    """通常のエッジ生成"""
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

def display_kwic(df_target, target_cols, search_words_list, filter_cols):
    """原文検索結果を表示 (複数単語AND検索・属性タグ付き)"""
    count = 0
    
    for i, row in df_target.iterrows():
        row_text = " ".join([str(row[c]) for c in target_cols if pd.notna(row[c])])
        
        # AND検索: リスト内の単語が「すべて」含まれているか確認
        if all(word in row_text for word in search_words_list):
            count += 1
            
            # ヒットした単語すべてをハイライト
            highlighted_text = row_text
            for word in search_words_list:
                highlighted_text = highlighted_text.replace(word, f"**{word}**")
            
            # 属性タグの作成
            tags = []
            for f_col in filter_cols:
                val = row[f_col]
                if pd.notna(val):
                    tags.append(f"[{val}]")
            tag_str = " ".join(tags)
            
            # 表示
            st.markdown(f"🏷️ **{tag_str}** : {highlighted_text}")
            st.markdown("---")
            
            if count >= 20:
                st.caption(f"※これ以上は省略します（他 {len(df_target)-count} 件の可能性あり）")
                break
    
    if count == 0:
        st.write("条件に一致する文章は見つかりませんでした。")

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
    
    # --- サイドバー: 設定エリア ---
    st.sidebar.title("⚙️ 設定パネル")
    
    # 1. 除外ワード設定
    with st.sidebar.expander("🚫 除外ワードの設定", expanded=True):
        st.caption("分析から外したい単語をスペース区切りで入力")
        new_words_input = st.text_input("追加", placeholder="例: 私　思う　アンケート")
        
        if new_words_input:
            words = new_words_input.replace('　', ' ').split()
            for w in words:
                if w not in st.session_state.user_stopwords:
                    st.session_state.user_stopwords.append(w)
            st.rerun()
        
        st.write(f"**現在の除外リスト ({len(st.session_state.user_stopwords)}語):**")
        st.text(", ".join(st.session_state.user_stopwords))
        
        if st.button("リストをリセット"):
            st.session_state.user_stopwords = []
            st.rerun()
            
    stop_words = DEFAULT_STOPWORDS + st.session_state.user_stopwords

    # 2. 全体フィルタリング設定
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 全体の絞り込み")
    
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

    # --- メインエリア ---
    
    mode = st.radio("表示モード", ["全体分析", "⚔️ 自由比較 (カスタム)"], horizontal=True)

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
            tab1, tab2, tab3, tab4 = st.tabs(["☁️ ワードクラウド", "🕸️ つながりマップ", "📈 ランキング", "🔎 原文検索"])
            
            with tab1:
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
                st.info("💡 **ヒント**: 同じ文脈でよく使われる単語同士が線で結ばれています。")
                c1, c2 = st.columns(2)
                net_top = c1.slider("表示単語数", 10, 150, 60)
                min_edge = c2.slider("最小の線の太さ", 1, 10, 2)
                
                sentences = []
                for i, row in df_filtered.iterrows():
                    row_text = " ".join([str(row[c]) for c in target_cols if pd.notna(row[c])])
                    sentences.append(row_text)

                tokens_list = [get_tokens(s, stop_words) for s in sentences]
                G = create_network(tokens_list, net_top, min_edge)
                
                if G.number_of_nodes() > 0:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    pos = nx.spring_layout(G, k=0.8, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#66b3ff', alpha=0.9, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
                    nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=10, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("つながりが見つかりません。")

            with tab3:
                c = Counter(tokens)
                words, counts = zip(*c.most_common(20))
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(words, counts, color='skyblue')
                ax.invert_yaxis()
                st.pyplot(fig)

            with tab4:
                st.markdown("#### 💬 複数の単語で検索 (AND検索)")
                st.caption("スペースで区切ると、それらすべてを含む文章を検索します。例: 「自分 価値」")
                input_str = st.text_input("検索したい単語", placeholder="例: 自分 価値")
                
                if input_str:
                    # 全角スペースを半角にして分割
                    search_words = input_str.replace('　', ' ').split()
                    st.markdown(f"**「{' + '.join(search_words)}」を含む回答一覧:**")
                    st.markdown("---")
                    display_kwic(df_filtered, target_cols, search_words, filter_candidates)

    # === B. 自由比較モード ===
    elif mode == "⚔️ 自由比較 (カスタム)":
        st.markdown("#### 条件を組み合わせてグループを作成")
        
        if not filter_candidates:
            st.error("比較できる属性列が見つかりません。")
        else:
            col_a_setup, col_b_setup = st.columns(2)
            
            with col_a_setup:
                st.info("🟦 **グループA の条件**")
                df_a = df_filtered.copy()
                with st.expander("条件を選択", expanded=True):
                    for col in filter_candidates:
                        vals = sorted(df[col].dropna().unique().tolist())
                        selected_a = st.multiselect(f"{col} (A)", vals, key=f"sel_a_{col}")
                        if selected_a:
                            df_a = df_a[df_a[col].isin(selected_a)]
                st.write(f"**人数:** {len(df_a)} 人")

            with col_b_setup:
                st.error("🟥 **グループB の条件**")
                df_b = df_filtered.copy()
                with st.expander("条件を選択", expanded=True):
                    for col in filter_candidates:
                        vals = sorted(df[col].dropna().unique().tolist())
                        selected_b = st.multiselect(f"{col} (B)", vals, key=f"sel_b_{col}")
                        if selected_b:
                            df_b = df_b[df_b[col].isin(selected_b)]
                st.write(f"**人数:** {len(df_b)} 人")

            if len(df_a) == 0 or len(df_b) == 0:
                st.warning("条件に該当するデータが0件です。")
            else:
                def get_combined_tokens(d):
                    txt = ""
                    for c in target_cols:
                        txt += " " + " ".join(d[c].dropna().astype(str).tolist())
                    return get_tokens(txt, stop_words)

                tokens_a = get_combined_tokens(df_a)
                tokens_b = get_combined_tokens(df_b)

                comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["☁️ ワードクラウド", "🕸️ 違いのネットワーク", "🦋 対比ランキング", "🔎 原文検索"])

                with comp_tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        if tokens_a:
                            wc_a = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_a))
                            fig, ax = plt.subplots()
                            ax.imshow(wc_a, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                    with c2:
                        if tokens_b:
                            wc_b = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_b))
                            fig, ax = plt.subplots()
                            ax.imshow(wc_b, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)

                with comp_tab2:
                    st.markdown("##### 🟦 青はAの特徴、🟥 赤はBの特徴、⬜ グレーは共通")
                    sentences_mixed = []
                    for i, row in df_a.iterrows():
                        sentences_mixed.append(" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]))
                    for i, row in df_b.iterrows():
                        sentences_mixed.append(" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]))
                    
                    tokens_list_mixed = [get_tokens(s, stop_words) for s in sentences_mixed]
                    G = create_network(tokens_list_mixed, top_n=60, min_edge=2)
                    
                    if G.number_of_nodes() > 0:
                        count_a = Counter(tokens_a)
                        count_b = Counter(tokens_b)
                        node_colors = []
                        for node in G.nodes():
                            fa = count_a.get(node, 0)
                            fb = count_b.get(node, 0)
                            total = fa + fb + 0.1 
                            ratio = fa / total
                            if ratio > 0.6: node_colors.append('#66b3ff')
                            elif ratio < 0.4: node_colors.append('#ff9999')
                            else: node_colors.append('#dddddd')
                        
                        fig, ax = plt.subplots(figsize=(9, 9))
                        pos = nx.spring_layout(G, k=0.7, seed=42)
                        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.9, ax=ax)
                        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, edge_color='gray', ax=ax)
                        nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=11, ax=ax)
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.warning("共通データ不足")

                with comp_tab3:
                    st.markdown("##### 🦋 バタフライチャート")
                    ca = Counter(tokens_a)
                    cb = Counter(tokens_b)
                    all_top_words = list(set([w for w, c in ca.most_common(15)] + [w for w, c in cb.most_common(15)]))
                    data = []
                    for w in all_top_words:
                        data.append({'word': w, 'A': ca.get(w, 0), 'B': cb.get(w, 0)})
                    df_comp = pd.DataFrame(data).sort_values('A', ascending=True)
                    if not df_comp.empty:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(df_comp['word'], -df_comp['A'], color='#66b3ff', label="グループA")
                        ax.barh(df_comp['word'], df_comp['B'], color='#ff9999', label="グループB")
                        ax.axvline(0, color='black', linewidth=0.8)
                        xticks = ax.get_xticks()
                        ax.set_xticklabels([str(abs(int(x))) for x in xticks])
                        ax.legend()
                        st.pyplot(fig)

                with comp_tab4:
                    st.markdown("#### 💬 文脈の違いを確認する (複数単語OK)")
                    input_str = st.text_input("検索したい単語 (スペース区切り)", placeholder="例: 授業 楽しい")
                    
                    if input_str:
                        search_words = input_str.replace('　', ' ').split()
                        col_res_a, col_res_b = st.columns(2)
                        with col_res_a:
                            st.info(f"🟦 グループAの検索結果")
                            display_kwic(df_a, target_cols, search_words, filter_candidates)
                        with col_res_b:
                            st.error(f"🟥 グループBの検索結果")
                            display_kwic(df_b, target_cols, search_words, filter_candidates)
