import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from janome.tokenizer import Tokenizer
from collections import Counter
from wordcloud import WordCloud
import re
import time
import networkx as nx

# --- 1. アプリの基本設定 ---
st.set_page_config(
    page_title="Text Analytics App",
    page_icon="📊",
    layout="wide"
)

# セッションステートの初期化（データを記憶させる金庫を作る）
if 'df' not in st.session_state:
    st.session_state.df = None

# ストップワードの定義
DEFAULT_STOPWORDS = [
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や",
    "れる", "など", "ない", "この", "ため", "その", "よう", "また", "もの",
    "ます", "です", "さん", "ちゃん", "くん"
]

# --- 2. 関数の定義 ---

@st.cache_data
def get_tokens(text, stop_words):
    """テキストから名詞・動詞・形容詞を抽出する関数"""
    t = Tokenizer()
    tokens = []
    if not isinstance(text, str):
        return []
    
    for token in t.tokenize(text):
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞', '形容詞'] and len(base) > 1 and base not in stop_words:
            tokens.append(base)
    return tokens

@st.cache_data
def create_demo_data():
    """デモ用のダミーデータを生成"""
    data = {
        '学年': ['1年', '1年', '2年', '2年', '3年', '3年', '1年', '2年', '3年', '1年'],
        '性別': ['男性', '女性', '男性', '女性', '男性', '女性', '女性', '男性', '女性', '男性'],
        '自由記述': [
            '野性味あふれる人材になりたいし、価値創造も重要だと思う。',
            '新しい価値を作るためには、野性的な勘が必要だと感じる。',
            '学校生活で野性味を磨き、社会で活躍したい。',
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

# --- 3. サイドバー（データ入力） ---
st.sidebar.title("🛠 設定 & データ入力")

# CSVアップロード機能を追加
input_method = st.sidebar.radio("データの読み込み方法", ["デモデータを使う", "スプレッドシートURL", "CSVファイルをアップロード"])

# --- データ読み込み処理 ---
# ボタンを押すと、session_state.df にデータが保存される仕組みに変更

if input_method == "デモデータを使う":
    if st.sidebar.button("デモデータをロード"):
        st.session_state.df = create_demo_data()
        st.sidebar.success("デモデータを読み込みました！")

elif input_method == "スプレッドシートURL":
    url = st.sidebar.text_input("スプレッドシートのURL")
    if st.sidebar.button("データを読み込む"):
        if url:
            try:
                with st.spinner("データ取得中..."):
                    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
                    if match:
                        file_id = match.group(1)
                        csv_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
                        st.session_state.df = pd.read_csv(csv_url)
                        st.sidebar.success(f"読み込み成功！ ({len(st.session_state.df)}行)")
                    else:
                        st.sidebar.error("URLの形式が正しくありません。")
            except Exception as e:
                st.sidebar.error(f"エラー: {e}")

elif input_method == "CSVファイルをアップロード":
    uploaded_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ", type=['csv'])
    if uploaded_file is not None:
        # アップロードされたらすぐに読み込む
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"読み込み成功！ ({len(st.session_state.df)}行)")
        except Exception as e:
            st.sidebar.error(f"エラー: {e}")

# --- 4. メイン画面 ---
st.title("📊 テキスト分析アプリ")

# セッションステート（金庫）にデータがあるか確認
if st.session_state.df is not None:
    df = st.session_state.df  # 使いやすいように変数に入れる
    
    # タブ作成
    tab1, tab2, tab3 = st.tabs(["📂 データセット確認", "📈 頻出単語分析", "☁️ ワードクラウド"])

    with tab1:
        st.header("読み込んだデータ")
        st.dataframe(df)

    with tab2:
        st.header("頻出単語ランキング")
        # すべての列を候補にする（数値データも選べるように修正）
        all_cols = df.columns
        target_col = st.selectbox("分析する文章の列を選んでください", all_cols, index=len(all_cols)-1)
        
        top_n = st.slider("表示する単語数", 5, 50, 10)

        if st.button("グラフを表示"):
            with st.spinner("解析中..."):
                # 選んだ列を強制的に文字型(str)に変換して結合
                text_data = " ".join(df[target_col].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                
                if tokens:
                    counter = Counter(tokens)
                    words, counts = zip(*counter.most_common(top_n))
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(words, counts, color='skyblue')
                    ax.invert_yaxis()
                    ax.set_title(f"「{target_col}」の頻出単語 TOP{top_n}")
                    st.pyplot(fig)
                else:
                    st.warning("分析可能な単語が見つかりませんでした。")

    with tab3:
        st.header("ワードクラウド")
        target_col_wc = st.selectbox("ワードクラウドにする列", all_cols, key='wc_select')
        
        if st.button("ワードクラウド作成"):
            with st.spinner("描画中..."):
                text_data = " ".join(df[target_col_wc].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                text_space_sep = " ".join(tokens)
                
                try:
                    wc = WordCloud(
                        background_color="white",
                        width=800, height=500,
                        regexp=r"[\w']+",
                        font_path="IPAexGothic.ttf"
                    ).generate(text_space_sep)
                    
                    fig_wc, ax_wc = plt.subplots(figsize=(12, 8))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                except Exception as e:
                    st.error("エラーが発生しました。フォントファイルがあるか確認してください。")
                    st.write(e)

else:
    st.info("👈 左のサイドバーからデータを読み込んでください")
