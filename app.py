import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Sentimen ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin: 10px 0;
    }
    .insight-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #1a237e;
    }
    h2, h3 {
        color: #283593;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_chatgpt_sentiment_all.csv')
        df = df.dropna(subset=['clean_comment', 'sentiment_label'])
        df['clean_comment'] = df['clean_comment'].astype(str)
        return df
    except FileNotFoundError:
        return None

# --- HEADER ---
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🤖 Dashboard Analisis Sentimen ChatGPT")
    st.markdown("**Analisis opini publik Indonesia terhadap ChatGPT berdasarkan data YouTube**")
with col_header2:
    st.image("https://img.icons8.com/fluency/96/chatgpt.png", width=80)

st.markdown("---")

# Load Data
df = load_data()

if df is not None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
        st.header("⚙️ Pengaturan Filter")
        
        # Filter Sentimen
        all_sentiments = df['sentiment_label'].unique().tolist()
        selected_sentiment = st.multiselect(
            "📊 Pilih Sentimen:",
            options=all_sentiments,
            default=all_sentiments
        )
        
        st.markdown("---")
        
        # Info Dataset
        st.subheader("📋 Info Dataset")
        st.info(f"**Total Data:** {len(df)} komentar")
        st.info(f"**Sumber:** YouTube Comments")
        
        st.markdown("---")
        st.caption("Dashboard by Streamlit")

    # Terapkan Filter
    df_filtered = df[df['sentiment_label'].isin(selected_sentiment)] if selected_sentiment else df

    # --- KPI METRICS ---
    st.subheader("📊 Ringkasan Statistik")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Komentar", 
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered)-len(df)} dari total" if len(df_filtered) != len(df) else None
        )
    
    with col2:
        if not df_filtered.empty:
            avg_score = df_filtered['sentiment_score'].mean()
            st.metric("Rata-rata Skor", f"{avg_score:.4f}", 
                     delta="Positif" if avg_score > 0 else "Negatif")
        else:
            st.metric("Rata-rata Skor", "0")
            
    with col3:
        if not df_filtered.empty:
            dom_sent = df_filtered['sentiment_label'].mode()[0]
            count_dom = len(df_filtered[df_filtered['sentiment_label'] == dom_sent])
            st.metric("Sentimen Dominan", dom_sent, delta=f"{count_dom} komentar")
        else:
            st.metric("Sentimen Dominan", "-")
    

    st.markdown("---")

    # --- INSIGHT BOX ---
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("### 💡 Insight Utama")
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        if not df_filtered.empty:
            pos_pct = (len(df_filtered[df_filtered['sentiment_label'] == 'Positif']) / len(df_filtered)) * 100
            neg_pct = (len(df_filtered[df_filtered['sentiment_label'] == 'Negatif']) / len(df_filtered)) * 100
            net_pct = (len(df_filtered[df_filtered['sentiment_label'] == 'Netral']) / len(df_filtered)) * 100
            
            st.markdown(f"""
            - **Sentimen Positif:** {pos_pct:.1f}% 
            - **Sentimen Negatif:** {neg_pct:.1f}%
            - **Sentimen Netral:** {net_pct:.1f}%
            """)
    
    with col_i2:
        avg_score = df_filtered['sentiment_score'].mean()
        if avg_score > 0:
            st.success("✅ Opini publik cenderung **POSITIF** terhadap ChatGPT")
        elif avg_score < 0:
            st.error("⚠️ Opini publik cenderung **NEGATIF** terhadap ChatGPT")
        else:
            st.info("➖ Opini publik **NETRAL** terhadap ChatGPT")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribusi Sentimen", 
        "☁️ Analisis Kata", 
        "🎯 Tren & Fenomena",
        "🔍 Kategorisasi Tema",
        "📄 Data & Etika"
    ])

    # === TAB 1: DISTRIBUSI SENTIMEN ===
    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🥧 Proporsi Sentimen")
            if not df_filtered.empty:
                sentiment_counts = df_filtered['sentiment_label'].value_counts()
                
                # Plotly Pie Chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=.4,
                    marker_colors=['#66b3ff', '#99ff99', '#ff9999']
                )])
                fig_pie.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Data tidak tersedia untuk filter ini.")

        with col_b:
            st.markdown("#### 📊 Distribusi Skor Sentimen")
            if not df_filtered.empty:
                fig_hist = px.histogram(
                    df_filtered, 
                    x='sentiment_score', 
                    nbins=30,
                    color_discrete_sequence=['#9370DB']
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                                  annotation_text="Netral (0)")
                fig_hist.update_layout(
                    xaxis_title="Skor (Negatif < 0 < Positif)",
                    yaxis_title="Frekuensi",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Data tidak tersedia.")
        
        # Bar Chart Sentimen
        st.markdown("#### 📊 Jumlah Komentar per Sentimen")
        if not df_filtered.empty:
            sentiment_counts = df_filtered['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            
            fig_bar = px.bar(
                sentiment_counts, 
                x='Sentimen', 
                y='Jumlah',
                color='Sentimen',
                color_discrete_map={'Positif': '#66b3ff', 'Netral': '#99ff99', 'Negatif': '#ff9999'},
                text='Jumlah'
            )
            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    # === TAB 2: WORD CLOUD ===
    with tab2:
        st.markdown("### ☁️ Visualisasi Kata Populer")
        
        all_text = " ".join(df_filtered['clean_comment'])
        
        if len(all_text) > 0:
            col_wc1, col_wc2 = st.columns([2, 1])
            
            with col_wc1:
                st.markdown("#### Word Cloud Keseluruhan")
                wordcloud = WordCloud(
                    width=1000, 
                    height=500, 
                    background_color='white', 
                    colormap='viridis',
                    max_words=100
                ).generate(all_text)
                
                fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            
            with col_wc2:
                st.markdown("#### 🔝 Top 10 Kata")
                words = all_text.split()
                word_counts = Counter(words).most_common(10)
                df_words = pd.DataFrame(word_counts, columns=['Kata', 'Frekuensi'])
                
                fig_word_bar = px.bar(
                    df_words, 
                    x='Frekuensi', 
                    y='Kata',
                    orientation='h',
                    color='Frekuensi',
                    color_continuous_scale='Viridis'
                )
                fig_word_bar.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_word_bar, use_container_width=True)
            
            # Word Cloud per Sentimen
            st.markdown("---")
            st.markdown("### ☁️ Word Cloud per Kategori Sentimen")
            col_s1, col_s2, col_s3 = st.columns(3)
            
            for idx, (col, sentiment, color) in enumerate(zip(
                [col_s1, col_s2, col_s3],
                ['Positif', 'Netral', 'Negatif'],
                ['Greens', 'Greys', 'Reds']
            )):
                with col:
                    st.markdown(f"#### {sentiment}")
                    sent_text = " ".join(df[df['sentiment_label'] == sentiment]['clean_comment'])
                    if len(sent_text) > 0:
                        wc_sent = WordCloud(
                            width=400, 
                            height=300, 
                            background_color='white',
                            colormap=color,
                            max_words=50
                        ).generate(sent_text)
                        
                        fig_s, ax_s = plt.subplots(figsize=(5, 4))
                        ax_s.imshow(wc_sent, interpolation='bilinear')
                        ax_s.axis('off')
                        st.pyplot(fig_s)
                    else:
                        st.info(f"Tidak ada data {sentiment}")
        else:
            st.warning("Tidak ada cukup teks untuk membuat Word Cloud.")

    # === TAB 3: TREN & FENOMENA ===
    with tab3:
        st.markdown("### 📈 Analisis Tren Lanjutan")

        # Statistik
        st.markdown("#### 📊 Tren Statistik")
        
        avg_score = df['sentiment_score'].mean()
        median_score = df['sentiment_score'].median()
        pos_pct = (len(df[df['sentiment_label'] == 'Positif']) / len(df)) * 100
        neg_pct = (len(df[df['sentiment_label'] == 'Negatif']) / len(df)) * 100
        
        st.markdown(f"""
        Data menunjukkan **rata-rata skor sentimen sebesar {avg_score:.4f}**. Angka positif ini menunjukkan bahwa 
        kehadiran ChatGPT di Indonesia disambut dengan tangan terbuka. Meskipun mayoritas komentar bersifat netral 
        (ditunjukkan oleh Median {median_score:.1f}), tarikan sentimen positif ({pos_pct:.1f}%) terbukti lebih dominan 
        dibandingkan sentimen negatif ({neg_pct:.1f}%), sehingga mengangkat rata-rata skor menjadi di atas nol.
        
        **Kesimpulan:** Mayoritas masih mempelajari (netral), namun arah opini publik secara umum bergerak ke arah 
        penerimaan positif, bukan penolakan.
        """)

        st.markdown("---")
        
        # Fenomena Sosial
        st.markdown("#### 🌍 Fenomena Sosial: Pragmatisme Digital")
        
        st.markdown("""
        Tren positif ini mencerminkan fenomena sosial yang bisa disebut sebagai **"Pragmatisme Digital"**. 
        Pengguna internet di Indonesia cenderung fokus pada nilai guna (utility) secara instan. 
        
        Hal ini bisa dilihat dari kata kunci sentimen positif yang dominan:
        - 🎯 **"bantu"** - ChatGPT sebagai asisten
        - 📝 **"tugas"** - Membantu pekerjaan akademis
        - ⚡ **"mudah"** - Kemudahan akses dan penggunaan
        - 🚀 **"cepat"** - Efisiensi waktu
        
        **Insight:** Ketakutan akan AI (penggantian kerja) memang ada, tetapi kalah kuat dibandingkan euforia 
        akan kemudahan menyelesaikan tugas kuliah atau pekerjaan sehari-hari. Publik Indonesia lebih melihat 
        ChatGPT sebagai **"asisten instan"** daripada **"ancaman masa depan"**.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # === TAB 4: KATEGORISASI TEMA ===
    with tab4:
        st.markdown("### 🎯 Kategorisasi Tema Utama")
        
        # Tabel Tema
        tema_data = {
            'Tema Besar': ['Produktivitas (Positif)', 'Kecemasan Karir (Negatif)'],
            'Kata Kunci': ['bantu, mudah, cepat, tugas', 'ganti, kerja, manusia, takut, pengangguran'],
            'Analisis': [
                'Netizen melihat ChatGPT sebagai alat yang memudahkan pekerjaan akademis dan kantor',
                'Ada tren ketakutan bahwa AI akan menggantikan peran manusia di dunia kerja'
            ]
        }
        
        df_tema = pd.DataFrame(tema_data)
        st.dataframe(df_tema, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visualisasi Tema
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("#### ✅ Tema Positif: Produktivitas")
            st.success("""
            **Karakteristik:**
            - Fokus pada manfaat praktis
            - Penyelesaian tugas lebih cepat
            - Efisiensi kerja dan belajar
            - Akses informasi mudah
            """)
        
        with col_t2:
            st.markdown("#### ⚠️ Tema Negatif: Kecemasan Karir")
            st.error("""
            **Karakteristik:**
            - Takut kehilangan pekerjaan
            - Kekhawatiran penggantian manusia
            - Isu pengangguran masa depan
            - Ketidakpastian karir
            """)
        
        # Evaluasi Data
        st.markdown("---")
        st.markdown("### 🔍 Evaluasi Kualitas Data")
        
        with st.expander("📊 Isu Representativitas Data"):
            st.warning("""
            **Bias Demografi:**
            
            Data yang digunakan diambil dari platform YouTube, sehingga memiliki kecenderungan bias demografi 
            ke arah segmen masyarakat yang menggunakan YouTube. Sampel ini kemungkinan besar didominasi oleh:
            - Pelajar dan mahasiswa
            - Pekerja kantoran
            - Penggiat teknologi yang aktif mencari informasi tentang AI
            
            **Keterbatasan:**
            
            Data ini **tidak merepresentasikan** suara dari kelompok masyarakat grassroot yang minim akses digital, 
            seperti petani, nelayan, pedagang pasar tradisional, atau generasi lanjut usia yang tidak mengonsumsi 
            konten ulasan teknologi di YouTube.
            """)
        
        with st.expander("🎥 Dominasi Topik"):
            st.info("""
            **Bias Kontekstual:**
            
            Dataset diambil dari satu video YouTube berjudul **"ChatGPT dan Masa Depan Pekerjaan Kita"** yang 
            menyebabkan adanya bias kontekstual yang kuat karena seluruh komentar berasal dari video yang secara 
            spesifik membahas topik "ChatGPT dan Masa Depan Pekerjaan".
            
            **Dampak:**
            
            Diskusi di kolom komentar sangat terarah pada isu pengaruh ChatGPT di ketenagakerjaan. Hal ini bisa 
            dilihat dari Word Cloud sentimen negatif dimana kata kunci seperti "kerja", "ganti", "manusia", dan 
            "pengangguran" mendominasi.
            """)

    # === TAB 5: DATA & ETIKA ===
    with tab5:
        st.markdown("### 📄 Data Detail & Analisis Etika")
        
        # Data Table
        st.markdown("#### 📋 Sample Data")
        st.dataframe(
            df_filtered[['Comment', 'clean_comment', 'sentiment_label', 'sentiment_score']].head(100),
            use_container_width=True
        )
        
        # Download Button
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name='sentiment_data_filtered.csv',
            mime='text/csv',
        )
        
        st.markdown("---")
        
        # Analisis Etika
        st.markdown("### 🔐 Analisis Keamanan dan Etika")
        
        with st.expander("📜 Sifat Data dan Izin Akses"):
            st.info("""
            **Ranah Publik (Open Domain):**
            
            Data yang digunakan dalam penelitian ini bersumber dari kolom komentar platform YouTube yang 
            dikategorikan sebagai data Ranah Publik. Secara etika penelitian data digital, pengambilan data 
            (web scraping) pada komentar YouTube **diperbolehkan** karena pengguna platform telah menyetujui 
            kebijakan layanan bahwa komentar yang mereka unggah bersifat publik dan dapat diakses oleh siapa saja.
            """)
        
        with st.expander("🔒 Perlindungan Privasi dan Anonimitas"):
            st.success("""
            **Teknik Anonimisasi:**
            
            Meskipun data bersifat publik, penelitian ini tetap menerapkan prinsip kehati-hatian dalam menjaga 
            privasi pengguna (user privacy). Identitas spesifik pengguna tidak relevan dengan tujuan analisis 
            sentimen umum.
            
            **Langkah Perlindungan:**
            - Fungsi yang menghapus pola username atau mention
            - Tidak menampilkan informasi identitas pengguna
            - Fokus pada analisis agregat, bukan individu
            """)
        
        with st.expander("⚠️ Risiko Penyalahgunaan"):
            st.error("""
            **Peringatan Penting:**
            
            Hasil analisis sentimen, meskipun bermanfaat untuk memetakan opini publik, memiliki risiko 
            penyalahgunaan jika diterapkan pada konteks yang salah.
            
            **Contoh Penyalahgunaan:**
            - **Diskriminasi Algoritmik:** Perusahaan atau perekrut kerja tidak etis jika menggunakan jejak 
              digital berupa komentar negatif seseorang terhadap AI di YouTube sebagai dasar penolakan lamaran kerja
            - **Profiling Individu:** Data sentimen berisiko jika digunakan sebagai alat profil individu untuk 
              pengambilan keputusan krusial
            
            **Prinsip Penggunaan:**
            
            Opini seseorang terhadap teknologi adalah preferensi pribadi yang dinamis dan tidak mencerminkan 
            kompetensi profesional. Oleh karena itu, hasil penelitian ini harus dibaca sebagai **gambaran 
            statistik populasi**, bukan penilaian karakter individu.
            """)

else:
    st.error("❌ File 'dataset_chatgpt_sentiment_all.csv' tidak ditemukan!")
    st.info("💡 Pastikan file CSV ada di folder yang sama dengan script ini.")