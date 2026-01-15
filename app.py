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

# --- 1. アプリの設定 ---
st.set_page_config(page_title="Text Analytics V9", layout="wide")

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
    
    # 除外ワードをセット（検索高速化）
    stop_set = set(stop_words)
    japanese_pattern = re.compile(r'[ぁ-んァ-ン一-龥]')
    
    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        # 除外判定
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

def create_colored_network(tokens_a, tokens_b, top_n, min_edge):
    """比較用ネットワーク (色分け機能付き)"""
    # 両方の単語カウント
    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)
    
    # 結合してネットワークを作る
    all_tokens_list = [tokens_a, tokens_b] # 簡易的に2文書として扱うとエッジが弱くなるため工夫が必要
    # 実際には文書ごとのリストが必要だが、ここでは簡易化のため「頻出語リスト」からグラフを作る
    
    # 上位語を抽出
    common_words = set([w for w, c in count_a.most_common(top_n)] + [w for w, c in count_b.most_common(top_n)])
    
    # エッジの生成（共起）は「元の文脈」が必要だが、
    # ここでは計算負荷を下げるため、簡易的にノードの色分けに注力する
    # ノードだけ定義して、色は「どちらに多く出ているか」で決める
    
    G = nx.Graph()
    
    # ノードの追加と色決定
    node_colors = []
    for word in common_words:
        freq_a = count_a.get(word, 0)
        freq_b = count_b.get(word, 0)
        total = freq_a + freq_b
        
        if total == 0: continue
        
        G.add_node(word, size=total)
        
        # 色分けロジック
        ratio = freq_a / total
        if ratio > 0.7:
            color = "#66b3ff" # 青 (A寄り)
        elif ratio < 0.3:
            color = "#ff9999" # 赤 (B寄り)
        else:
            color = "#dddddd" # グレー (共通)
        
        # 属性として保存 (描画時にリスト化するため)
        G.nodes[word]['color'] = color
        
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
    
    # --- サイドバー: 設定エリア ---
    st.sidebar.title("⚙️ 設定パネル")
    
    # 1. 除外ワード設定
    with st.sidebar.expander("🚫 除外ワードの設定", expanded=True):
        st.caption("分析から外したい単語をスペース区切りで入力")
        new_words_input = st.text_input("追加", placeholder="例: 私　思う　アンケート")
        
        if new_words_input:
            # 全角スペースを半角に変換して分割
            words = new_words_input.replace('　', ' ').split()
            added_count = 0
            for w in words:
                if w not in st.session_state.user_stopwords:
                    st.session_state.user_stopwords.append(w)
                    added_count += 1
            if added_count > 0:
                st.success(f"{added_count}語を追加しました")
                time.sleep(1) # 追加したことがわかるように少し待つ
                st.rerun()
        
        # 除外リストの表示（削除機能付きは複雑になるので、リセットのみ実装）
        st.write(f"**現在の除外リスト ({len(st.session_state.user_stopwords)}語):**")
        st.text(", ".join(st.session_state.user_stopwords))
        
        if st.button("リストをリセット"):
            st.session_state.user_stopwords = []
            st.rerun()
            
    stop_words = DEFAULT_STOPWORDS + st.session_state.user_stopwords

    # 2. フィルタリング設定
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
    
    # モード切替
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
            st.warning("表示できる単語がありません。除外ワードを減らすか、データを増やしてください。")
        else:
            tab1, tab2, tab3 = st.tabs(["☁️ ワードクラウド", "🕸️ つながりマップ", "📈 ランキング"])
            
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
                st.info("💡 **ヒント**: 頻繁にセットで登場する単語同士が線で結ばれています。")
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

    # === B. グループ比較モード (V9強化版) ===
    elif mode == "⚔️ グループ比較":
        st.markdown("#### 2つのグループの違いを一画面で比較")
        
        if not filter_candidates:
            st.error("比較できる属性列が見つかりません。")
        else:
            col_comp_1, col_comp_2, col_comp_3 = st.columns([1, 1, 1])
            target_attr = col_comp_1.selectbox("比較する項目", filter_candidates)
            
            unique_vals = sorted(df_filtered[target_attr].dropna().unique().tolist())
            
            # 複数選択 (Multiselect) に変更
            vals_a = col_comp_2.multiselect("グループA (青)", unique_vals, default=[unique_vals[0]] if unique_vals else None)
            
            # デフォルトでA以外を選択状態にする工夫
            default_b = [v for v in unique_vals if v not in vals_a]
            if not default_b and unique_vals: default_b = [unique_vals[-1]]
            
            vals_b = col_comp_3.multiselect("グループB (赤)", unique_vals, default=default_b)

            if not vals_a or not vals_b:
                st.warning("比較するグループを選択してください。")
            else:
                # データ分割 & トークン化
                df_a = df_filtered[df_filtered[target_attr].isin(vals_a)]
                df_b = df_filtered[df_filtered[target_attr].isin(vals_b)]
                
                def get_combined_tokens(d):
                    txt = ""
                    for c in target_cols:
                        txt += " " + " ".join(d[c].dropna().astype(str).tolist())
                    return get_tokens(txt, stop_words)

                tokens_a = get_combined_tokens(df_a)
                tokens_b = get_combined_tokens(df_b)

                st.markdown(f"**分析対象数:** 🟦 グループA: {len(df_a)}件 vs 🟥 グループB: {len(df_b)}件")

                # タブでグラフを切り替え
                comp_tab1, comp_tab2, comp_tab3 = st.tabs(["☁️ ワードクラウド", "🕸️ 違いのネットワーク", "🦋 対比ランキング"])

                with comp_tab1:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info("🟦 グループA の特徴")
                        if tokens_a:
                            wc_a = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_a))
                            fig, ax = plt.subplots()
                            ax.imshow(wc_a, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                    with c2:
                        st.error("🟥 グループB の特徴")
                        if tokens_b:
                            wc_b = WordCloud(background_color="white", width=400, height=300, font_path="IPAexGothic.ttf").generate(" ".join(tokens_b))
                            fig, ax = plt.subplots()
                            ax.imshow(wc_b, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)

                with comp_tab2:
                    st.markdown("##### 🟦 青はAによく出る言葉、🟥 赤はBによく出る言葉")
                    # 簡易的に結合したネットワークを描画し、ノードの色を変える
                    
                    # 共起計算用に一旦sentencesを作る
                    sentences_mixed = []
                    # Aの文
                    for i, row in df_a.iterrows():
                        sentences_mixed.append(" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]))
                    # Bの文
                    for i, row in df_b.iterrows():
                        sentences_mixed.append(" ".join([str(row[c]) for c in target_cols if pd.notna(row[c])]))
                    
                    tokens_list_mixed = [get_tokens(s, stop_words) for s in sentences_mixed]
                    
                    # ネットワーク生成
                    G = create_network(tokens_list_mixed, top_n=60, min_edge=2)
                    
                    if G.number_of_nodes() > 0:
                        # 色分け計算
                        count_a = Counter(tokens_a)
                        count_b = Counter(tokens_b)
                        
                        node_colors = []
                        for node in G.nodes():
                            fa = count_a.get(node, 0)
                            fb = count_b.get(node, 0)
                            total = fa + fb + 0.1 # ゼロ除算防止
                            ratio = fa / total
                            
                            if ratio > 0.6:
                                node_colors.append('#66b3ff') # A寄り(青)
                            elif ratio < 0.4:
                                node_colors.append('#ff9999') # B寄り(赤)
                            else:
                                node_colors.append('#dddddd') # 共通(グレー)
                        
                        fig, ax = plt.subplots(figsize=(9, 9))
                        pos = nx.spring_layout(G, k=0.7, seed=42)
                        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.9, ax=ax)
                        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.4, edge_color='gray', ax=ax)
                        nx.draw_networkx_labels(G, pos, font_family='IPAexGothic', font_size=11, ax=ax)
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.warning("共通するつながりが少なすぎて描画できません。")

                with comp_tab3:
                    st.markdown("##### 🦋 バタフライチャート (左右の頻度比較)")
                    # 両方のトップ20単語を取得してマージ
                    ca = Counter(tokens_a)
                    cb = Counter(tokens_b)
                    
                    # AとBあわせた上位単語
                    all_top_words = list(set([w for w, c in ca.most_common(15)] + [w for w, c in cb.most_common(15)]))
                    
                    data = []
                    for w in all_top_words:
                        data.append({'word': w, 'A': ca.get(w, 0), 'B': cb.get(w, 0)})
                    
                    df_comp = pd.DataFrame(data).sort_values('A', ascending=True) # Aの順でソート
                    
                    if not df_comp.empty:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Aは左（マイナス方向）に伸ばす
                        ax.barh(df_comp['word'], -df_comp['A'], color='#66b3ff', label=f"グループA ({len(df_a)})")
                        # Bは右（プラス方向）に伸ばす
                        ax.barh(df_comp['word'], df_comp['B'], color='#ff9999', label=f"グループB ({len(df_b)})")
                        
                        # 真ん中の線
                        ax.axvline(0, color='black', linewidth=0.8)
                        
                        # ラベル（マイナスをプラス表記に戻す）
                        xticks = ax.get_xticks()
                        ax.set_xticklabels([str(abs(int(x))) for x in xticks])
                        
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.write("データ不足")
