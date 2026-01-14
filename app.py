import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from janome.tokenizer import Tokenizer
from collections import Counter
from wordcloud import WordCloud
import re
import time

# --- 1. アプリの基本設定 ---
st.set_page_config(
    page_title="Text Analytics App",
    page_icon="📊",
    layout="wide"
)

# ストップワードの定義（お好みで追加可能）
DEFAULT_STOPWORDS = [
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
    "ある", "いる", "も", "する", "から", "な", "こと", "として", "い", "や",
    "れる", "など", "ない", "この", "ため", "その", "よう", "また", "もの",
    "ます", "です", "さん", "ちゃん", "くん"
]

# --- 2. 関数の定義（キャッシュを使って高速化） ---

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
    """デモ用のダミーデータを生成する関数"""
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

def generate_wordcloud(text, font_path=None):
    """ワードクラウドを生成する関数"""
    # MacやLinux(Streamlit Cloud)環境での文字化け対策としてフォント指定が必要な場合があります
    # 今回はjapanize_matplotlibのフォントパスを借用するか、デフォルトで試みます
    wc = WordCloud(
        background_color="white",
        width=800,
        height=500,
        font_path="IPAexGothic.ttf", # ※同じフォルダにフォントファイルがある場合
        regexp=r"[\w']+"
    ).generate(text)
    return wc

# --- 3. サイドバー（設定・入力エリア） ---
st.sidebar.title("🛠 設定 & データ入力")

input_method = st.sidebar.radio("データの読み込み方法", ["デモデータを使う", "スプレッドシートURLを入力"])

df = None

if input_method == "デモデータを使う":
    if st.sidebar.button("デモデータをロード"):
        with st.spinner("デモデータを生成中..."):
            time.sleep(1) # 処理感の演出
            df = create_demo_data()
            st.sidebar.success("デモデータを読み込みました！")

else:
    url = st.sidebar.text_input("スプレッドシートのURL")
    if st.sidebar.button("データを読み込む"):
        if url:
            try:
                with st.spinner("スプレッドシートからデータを取得中..."):
                    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
                    if match:
                        file_id = match.group(1)
                        csv_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
                        df = pd.read_csv(csv_url)
                        st.sidebar.success(f"読み込み成功！ ({len(df)}行)")
                    else:
                        st.sidebar.error("URLの形式が正しくありません。")
            except Exception as e:
                st.sidebar.error(f"エラーが発生しました: {e}")
                st.sidebar.info("ヒント: シートの共有設定が「リンクを知っている全員」になっているか確認してください。")

# --- 4. メイン画面の構築 ---
st.title("📊 テキスト分析アプリ")
st.markdown("アンケートなどの自由記述データを分析・可視化するツールです。")

if df is not None:
    # データが読み込まれている場合のみタブを表示
    tab1, tab2, tab3 = st.tabs(["📂 データセット確認", "📈 頻出単語分析", "☁️ ワードクラウド"])

    with tab1:
        st.header("読み込んだデータ")
        st.dataframe(df)

    with tab2:
        st.header("頻出単語ランキング")
        
        # 分析する列を選択
        text_cols = df.select_dtypes(include=['object']).columns
        target_col = st.selectbox("分析する列を選んでください", text_cols, index=len(text_cols)-1)
        
        # 表示件数のスライダー
        top_n = st.slider("表示する単語数", 5, 50, 10)

        if st.button("グラフを表示"):
            with st.spinner("テキスト解析中..."):
                # 全テキストを結合
                text_data = " ".join(df[target_col].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                
                if tokens:
                    counter = Counter(tokens)
                    words, counts = zip(*counter.most_common(top_n))
                    
                    # グラフ描画
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(words, counts, color='skyblue')
                    ax.invert_yaxis() # 上位を上に
                    ax.set_title(f"「{target_col}」の頻出単語 TOP{top_n}")
                    st.pyplot(fig)
                else:
                    st.warning("分析可能な単語が見つかりませんでした。")

    with tab3:
        st.header("ワードクラウド")
        target_col_wc = st.selectbox("ワードクラウドにする列", text_cols, key='wc_select')
        
        if st.button("ワードクラウド作成"):
            with st.spinner("描画中..."):
                text_data = " ".join(df[target_col_wc].dropna().astype(str).tolist())
                tokens = get_tokens(text_data, DEFAULT_STOPWORDS)
                text_space_sep = " ".join(tokens)
                
                try:
                    # フォントパスの問題を回避するための簡易try-except
                    # Streamlit Cloud等で日本語フォントがないと文字化けするため、
                    # 実際にはリポジトリに IPAexGothic.ttf などを置いて指定するのが確実です
                    wc = WordCloud(
                        background_color="white",
                        width=800, height=500,
                        regexp=r"[\w']+",
                        font_path="IPAexGothic.ttf" # フォントファイルがある前提
                    ).generate(text_space_sep)
                    
                    fig_wc, ax_wc = plt.subplots(figsize=(12, 8))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                except Exception as e:
                    st.error("ワードクラウド生成エラー（フォントファイルが見つからない可能性があります）")
                    st.write(e)

else:
    # データ未読み込み時の案内
    st.info("👈 左のサイドバーから「デモデータ」を選択して試すか、スプレッドシートのURLを入力してください。")
