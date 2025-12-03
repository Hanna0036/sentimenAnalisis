import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import string

# --- LIBRARY NLP & SENTIMEN ---
# Pastikan sudah install: pip install Sastrawi vaderSentiment
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Sentimen Otomatis", page_icon="🤖", layout="wide")

# --- DEFINISI KAMUS & STOPWORDS (Dari Notebook tele.ipynb) ---
key_norm = {
    "yg": "yang", "gan": "juragan", "n": "dan", "dgn": "dengan", "sdh": "sudah", "udh": "sudah",
    "bgt": "banget", "krn": "karena", "sy": "saya", "gw": "saya", "gue": "saya", "aku": "saya",
    "mw": "mau", "mkn": "makan", "bs": "bisa", "bisa": "dapat", "gak": "tidak", "ga": "tidak",
    "gk": "tidak", "tp": "tapi", "kalo": "kalau", "kl": "kalau", "dlm": "dalam", "tdk": "tidak",
    "jgn": "jangan", "maap": "maaf", "sori": "maaf", "ap": "apa", "apaan": "apa", "utk": "untuk",
    "ya": "iya", "sm": "sama", "u": "kamu", "lu": "kamu", "lo": "kamu", "dr": "dari", "aj": "saja",
    "aja": "saja", "kpn": "kapan", "km": "kamu", "trus": "terus", "jd": "jadi", "blm": "belum",
    "tlg": "tolong", "dg": "dengan", "adlh": "adalah", "spt": "seperti", "tsb": "tersebut",
    "scr": "secara", "thd": "terhadap", "pd": "pada", "ml": "machine learning", "ai": "artificial intelligence"
}

additional_stopwords = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "adalah",
    "saya", "kami", "kita", "mereka", "dia", "ia", "juga", "akan", "bisa", "ada",
    "atau", "serta", "sedang", "sudah", "telah", "bagi", "kepada", "antara", "oleh",
    "dengan", "sebagai", "sampai", "karena", "jika", "namun", "tetapi", "kah", "pun",
    "tentang", "seperti", "melalui", "sementara", "manakala", "sebab", "maka", "tersebut",
    "yaitu", "yakni", "daripada", "paling", "tanpa", "sesuatu", "segala", "seluruh",
    "dapat", "juragan", "banget", "mau", "tapi", "kalau", "dalam", "jangan", "maaf",
    "apa", "iya", "sama", "kamu", "saja", "kapan", "terus", "jadi", "belum", "tolong",
    "terhadap", "mas", "pak", "bu", "bang", "kak", "gan", "bro", "min", "admin", "bapak", "ibu",
    "tidak", "tak", "tiada", "bukan", "enggak", "nggak", "kagak", "ndak",
    "nih", "tuh", "deh", "dong", "kok", "sih", "kan", "lah", "mah", "yah"
}

# --- FUNGSI CLEANING (Dicache agar cepat saat reload) ---
@st.cache_resource
def load_nlp_tools():
    # Inisialisasi Sastrawi hanya sekali
    factory_sw = StopWordRemoverFactory()
    stopword_remover = factory_sw.create_stop_word_remover()
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    analyzer = SentimentIntensityAnalyzer()
    return stopword_remover, stemmer, analyzer

def clean_text(text, stopword_remover, stemmer):
    if not isinstance(text, str): return ""
    
    # 1. Basic Cleaning
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Slang Normalization
    words = text.split()
    words = [key_norm.get(w, w) for w in words]
    text = " ".join(words)
    
    # 3. Stopword Sastrawi
    text = stopword_remover.remove(text)
    
    # 4. Stopword Manual
    words = text.split()
    words = [w for w in words if w not in additional_stopwords]
    text = " ".join(words)
    
    # 5. Stemming (Proses ini yang paling lama)
    text = stemmer.stem(text)
    
    return text

# --- FUNGSI UTAMA: LOAD & PROCESS DATA ---
@st.cache_data(show_spinner=False)
def process_data(filename):
    stopword_remover, stemmer, analyzer = load_nlp_tools()
    
    try:
        df = pd.read_csv(filename, on_bad_lines='skip', engine='python')
        
        # Hapus duplikat awal
        df = df.drop_duplicates(subset=['Comment'])
        df = df.dropna(subset=['Comment'])
        
        # Preprocessing (Cleaning)
        # Kita pakai progress bar karena stemming Sastrawi bisa lama
        df['clean_comment'] = df['Comment'].apply(lambda x: clean_text(x, stopword_remover, stemmer))
        
        # Hapus data kosong setelah cleaning
        df = df[df['clean_comment'] != '']
        
        # Analisis Sentimen menggunakan VADER
        # VADER bekerja baik untuk teks pendek, meskipun optimalnya untuk bhs inggris, 
        # namun sering digunakan sebagai baseline untuk sentimen umum.
        # Jika ingin akurasi tinggi bahasa Indonesia, bisa ganti dengan library lain nanti.
        def get_vader_score(text):
            return analyzer.polarity_scores(text)['compound']

        df['sentiment_score'] = df['clean_comment'].apply(get_vader_score)
        
        def get_label(score):
            if score >= 0.05: return 'Positif'
            elif score <= -0.05: return 'Negatif'
            else: return 'Netral'
            
        df['sentiment_label'] = df['sentiment_score'].apply(get_label)
        
        return df
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

# --- MULAI TAMPILAN DASHBOARD ---

st.title("🤖 Dashboard Analisis Sentimen Otomatis")
st.markdown("""
Aplikasi ini akan membaca **data mentah**, membersihkannya (stemming, slang removal), 
dan melakukan analisis sentimen secara **real-time**.
""")

# Load Data Mentah
raw_file = 'dataset_chatgpt_scrap.csv'

with st.spinner('Sedang memproses data (Cleaning & Stemming)... Proses ini mungkin memakan waktu 1-2 menit untuk data baru.'):
    df = process_data(raw_file)

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Filter Hasil")
    selected_sentiment = st.sidebar.multiselect(
        "Tampilkan Sentimen:",
        options=['Positif', 'Netral', 'Negatif'],
        default=['Positif', 'Netral', 'Negatif']
    )
    
    if not selected_sentiment:
        st.warning("Silakan pilih minimal satu sentimen di sidebar.")
        df_filtered = df
    else:
        df_filtered = df[df['sentiment_label'].isin(selected_sentiment)]

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Komentar Bersih", len(df_filtered))
    col2.metric("Sentimen Dominan", df_filtered['sentiment_label'].mode()[0] if not df_filtered.empty else "-")
    col3.metric("Rata-rata Skor", f"{df_filtered['sentiment_score'].mean():.3f}" if not df_filtered.empty else "0")
    
    st.divider()

    # --- VISUALISASI ---
    tab1, tab2, tab3 = st.tabs(["📊 Analisis Sentimen", "☁️ Word Cloud", "📝 Data Hasil Olah"])

    with tab1:
        st.subheader("Distribusi Sentimen")
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Pie Chart
            counts = df_filtered['sentiment_label'].value_counts()
            if not counts.empty:
                fig1, ax1 = plt.subplots()
                colors = {'Positif': '#66b3ff', 'Netral': '#99ff99', 'Negatif': '#ff9999'}
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
                        colors=[colors.get(x, '#ccc') for x in counts.index])
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.info("Tidak ada data untuk ditampilkan.")
                
        with col_b:
            # Histogram Skor
            st.markdown("**Sebaran Skor Sentimen (VADER Compound)**")
            fig2, ax2 = plt.subplots()
            sns.histplot(df_filtered['sentiment_score'], bins=20, kde=True, ax=ax2, color='purple')
            ax2.axvline(0, color='red', linestyle='--')
            st.pyplot(fig2)

    with tab2:
        st.subheader("Analisis Kata (Word Cloud)")
        
        all_text = " ".join(df_filtered['clean_comment'])
        if len(all_text) > 0:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("Tidak cukup kata untuk membuat Word Cloud.")

    with tab3:
        st.subheader("Tabel Data Hasil Cleaning")
        st.dataframe(df_filtered[['Comment', 'clean_comment', 'sentiment_label', 'sentiment_score']])
        
else:
    st.error(f"File '{raw_file}' tidak ditemukan di folder ini. Silakan upload file csv mentah.")