import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from janome.tokenizer import Tokenizer
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
import itertools
import re
import os
import streamlit.components.v1 as components
from pyvis.network import Network

# --- フォントの設定（japanize_matplotlibの代わり） ---
font_path = "IPAexGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='IPAexGothic')
else:
    st.error(f"フォントファイル {font_path} が見つかりません。GitHubにアップロードされているか確認してください。")
# ライブラリの有無を確認
try:
    import community.community_louvain as community_louvain
except ImportError:
    community_louvain = None

# --- 1. アプリの設定 ---
st.set_page_config(page_title="Text Analytics V13", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'user_stopwords' not in st.session_state:
    st.session_state.user_stopwords = []

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

def create_network_interactive(tokens_list, top_n, min_edge):
    all_words = []
    for tokens in tokens_list: all_words.extend(tokens)
    word_counts = Counter(all_words)
    pair_list = []
    for tokens in tokens_list:
        if len(tokens) >= 2:
            pair_list.extend(itertools.combinations(sorted(tokens), 2))
    c = Counter(pair_list)
    top_pairs = c.most_common(top_n)
    G = nx.Graph()
    for (u, v), weight in top_pairs:
        if weight >= min_edge:
            G.add_edge(u, v, value=weight, title=f"共起回数: {weight}回")
            if u not in G.nodes:
                G.add_node(u, size=word_counts[u] * 1.5, title=f"{u}: {word_counts[u]}回", group=1)
            if v not in G.nodes:
                G.add_node(v, size=word_counts[v] * 1.5, title=f"{v}: {word_counts[v]}回", group=1)
    if community_louvain and G.number_of_nodes() > 0:
        try:
            partition = community_louvain.best_partition(G)
            for node, gid in partition.items(): G.nodes[node]['group'] = gid
        except: pass
    return G

def classify_columns(df):
    filter_cols, text_cols = [], []
    for col in df.columns:
        if df[col].nunique() < 50: filter_cols.append(col)
        elif df[col].dtype == 'object': text_cols.append(col)
    return filter_cols, text_cols

@st.cache_data
def get_tokens(text, stop_words):
    t = Tokenizer()
    tokens = []
    if not isinstance(text, str): return []
    stop_set = set(stop_words)
    pattern = re.compile(r'[ぁ-んァ-ン一-龥]')
    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if (pos in ['名詞', '動詞', '形容詞'] and len(base) > 1 and 
            base not in stop_set and pattern.search(base)):
            tokens.append(base)
    return tokens

@st.cache_data
def create_network(tokens_list, top_n, min_edge):
    pair_list = []
    for tokens in tokens_list:
        if len(tokens) >= 2: pair_list.extend(itertools.combinations(tokens, 2))
    c = Counter(pair_list)
    G = nx.Graph()
    for (u, v), weight in c.most_common(top_n):
        if weight >= min_edge: G.add_edge(u, v, weight=weight)
    return G

def display_kwic(df_target, target_cols, search_words_list, filter_cols):
    count = 0
    for i, row in df_target.iterrows():
        row_text = " ".join([str(row[c]) for c in target_cols if pd.notna(row[c])])
        if all(word in row_text for word in search_words_list):
            count += 1
            res = row_text
            for w in search_words_list: res = res.replace(w, f"**{w}**")
            tags = " ".join([f"[{row[f]}]" for f in filter_cols if pd.notna(row[f])])
            st.markdown(f"🏷️ **{tags}** : {res}")
            st.markdown("---")
            if count >= 20: break
    if count == 0: st.write("該当なし")

# --- 3. メイン ---

if st.session_state.df is None:
    st.title("📂 データの読み込み")
    uploaded_file = st.file_uploader("ファイルをアップロード", type=['csv', 'xlsx'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.rerun()
else:
    df = st.session_state.df
    filter_candidates, text_candidates = classify_columns(df)
    stop_words = DEFAULT_STOPWORDS + st.session_state.user_stopwords

    # サイドバー（省略せずに記載）
    st.sidebar.title("⚙️ 設定")
    with st.sidebar.expander("🚫 除外ワード"):
        new_in = st.text_input("追加（スペース区切り）")
        if new_in:
            for w in new_in.replace('　', ' ').split():
                if w not in st.session_state.user_stopwords: st.session_state.user_stopwords.append(w)
            st.rerun()
        if st.button("リセット"):
            st.session_state.user_stopwords = []
            st.rerun()

    df_filtered = df.copy()
    for col in filter_candidates:
        sel = st.sidebar.multiselect(f"{col}", sorted(df[col].dropna().unique().tolist()))
        if sel: df_filtered = df_filtered[df_filtered[col].isin(sel)]
    
    if st.sidebar.button("ファイルを変更"):
        st.session_state.df = None
        st.rerun()

    mode = st.radio("表示モード", ["全体分析", "⚔️ 自由比較 (カスタム)"], horizontal=True)
    target_cols = text_candidates if text_candidates else df.columns

    if len(df_filtered) == 0:
        st.error("データがありません")
    
    # === A. 全体分析モード ===
    elif mode == "全体分析":
        tab1, tab2, tab3, tab4 = st.tabs(["☁️ WC", "🕸️ つながり", "📈 頻出", "🔎 原文"])
        full_txt = " ".join(df_filtered[target_cols].astype(str).values.flatten())
        tokens = get_tokens(full_txt, stop_words)

        with tab1:
            try:
                wc = WordCloud(background_color="white", font_path="IPAexGothic.ttf").generate(" ".join(tokens))
                fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)
            except: st.error("フォント読込エラー")
        
        with tab2:
            st.markdown("#### 🕸️ つながりマップ")
            c1, c2 = st.columns(2)
            net_top = c1.slider("単語ペア数", 10, 200, 60, key="all_top")
            min_e = c2.slider("最小共起数", 1, 10, 2, key="all_min")
            sents = [" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]) for _, row in df_filtered.iterrows()]
            t_list = [get_tokens(s, stop_words) for s in sents]
            G = create_network_interactive(t_list, net_top, min_e)
            if G.number_of_nodes() > 0:
                net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                net.from_nx(G); net.force_atlas_2based(gravity=-50)
                path = "/tmp/pyvis_all.html" if os.path.exists("/tmp") else "pyvis_all.html"
                net.save_graph(path)
                with open(path, 'r', encoding='utf-8') as f: components.html(f.read(), height=610)
            else: st.warning("データ不足")

        with tab3:
            if tokens:
                c = Counter(tokens); w, cnt = zip(*c.most_common(20))
                fig, ax = plt.subplots(); ax.barh(w, cnt); ax.invert_yaxis(); st.pyplot(fig)

        with tab4:
            query = st.text_input("検索ワード（スペース区切り）", key="all_search")
            if query: display_kwic(df_filtered, target_cols, query.split(), filter_candidates)

    # === B. 自由比較モード（ここにご提示のロジックを統合しました） ===
    elif mode == "⚔️ 自由比較 (カスタム)":
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
                        if selected_a: df_a = df_a[df_a[col].isin(selected_a)]
                st.write(f"**人数:** {len(df_a)} 人")

            with col_b_setup:
                st.error("🟥 **グループB の条件**")
                df_b = df_filtered.copy()
                with st.expander("条件を選択", expanded=True):
                    for col in filter_candidates:
                        vals = sorted(df[col].dropna().unique().tolist())
                        selected_b = st.multiselect(f"{col} (B)", vals, key=f"sel_b_{col}")
                        if selected_b: df_b = df_b[df_b[col].isin(selected_b)]
                st.write(f"**人数:** {len(df_b)} 人")

            if len(df_a) == 0 or len(df_b) == 0:
                st.warning("各グループに少なくとも1人以上のデータが必要です。")
            else:
                def get_combined_tokens(d):
                    txt = " ".join(d[target_cols].astype(str).values.flatten())
                    return get_tokens(txt, stop_words)

                tokens_a = get_combined_tokens(df_a)
                tokens_b = get_combined_tokens(df_b)

                comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs(["☁️ WC", "🕸️ 違いのネット", "🦋 対比", "🔎 原文"])

                with comp_tab1:
                    c1, c2 = st.columns(2)
                    for t, col, title in [(tokens_a, c1, "A"), (tokens_b, c2, "B")]:
                        with col:
                            if t:
                                wc = WordCloud(background_color="white", font_path="IPAexGothic.ttf").generate(" ".join(t))
                                fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)

                with comp_tab2:
                    st.markdown("##### 🟦 青はA、🟥 赤はB、⬜ グレーは共通")
                    sents_mixed = [" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]) for _, row in pd.concat([df_a, df_b]).iterrows()]
                    t_list_m = [get_tokens(s, stop_words) for s in sents_mixed]
                    G = create_network(t_list_m, top_n=60, min_edge=2)
                    if G.number_of_nodes() > 0:
                        ca, cb = Counter(tokens_a), Counter(tokens_b)
                        colors = []
                        for node in G.nodes():
                            fa, fb = ca.get(node, 0), cb.get(node, 0)
                            ratio = fa / (fa + fb + 0.1)
                            if ratio > 0.6: colors.append('#66b3ff')
                            elif ratio < 0.4: colors.append('#ff9999')
                            else: colors.append('#dddddd')
                        fig, ax = plt.subplots(figsize=(9, 9))
                        pos = nx.spring_layout(G, k=0.7, seed=42)
                        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=colors, alpha=0.9, ax=ax)
                        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
                        nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=11, ax=ax)
                        ax.axis('off'); st.pyplot(fig)
                    else: st.warning("共通データ不足")

                with comp_tab3:
                    st.markdown("##### 🦋 バタフライチャート")
                    ca, cb = Counter(tokens_a), Counter(tokens_b)
                    all_top = list(set([w for w, _ in ca.most_common(15)] + [w for w, _ in cb.most_common(15)]))
                    df_c = pd.DataFrame([{'w': w, 'A': ca.get(w, 0), 'B': cb.get(w, 0)} for w in all_top]).sort_values('A')
                    if not df_c.empty:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(df_c['w'], -df_c['A'], color='#66b3ff', label="A")
                        ax.barh(df_c['w'], df_c['B'], color='#ff9999', label="B")
                        ax.axvline(0, color='black'); ax.legend(); st.pyplot(fig)

                with comp_tab4:
                    query = st.text_input("比較検索ワード", key="comp_search")
                    if query:
                        c1, c2 = st.columns(2)
                        with c1: display_kwic(df_a, target_cols, query.split(), filter_candidates)
                        with c2: display_kwic(df_b, target_cols, query.split(), filter_candidates)
