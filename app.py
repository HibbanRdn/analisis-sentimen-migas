import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import re
import datetime
import matplotlib.pyplot as plt

# ==========================================================
# === KONFIGURASI HALAMAN ==================================
# ==========================================================
st.set_page_config(
    page_title="Analisis Sentimen YouTube",
    page_icon="🎥",
    layout="wide"
)

# ==========================================================
# === FUNGSI PRE-PROCESSING TEKS ===========================
# ==========================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Case folding
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Hapus URL
    text = re.sub(r'\@\w+|\#','', text) # Hapus mention dan hashtag
    text = re.sub(r'[^\w\s]', '', text) # Hapus tanda baca
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = text.strip()
    return text

# ==========================================================
# === JUDUL APLIKASI =======================================
# ==========================================================
st.markdown(
    "<h1 style='text-align:left; margin-bottom:0;'>🎥 Analisis Sentimen Komentar YouTube</h1>",
    unsafe_allow_html=True
)
st.write("""
Aplikasi ini memprediksi sentimen (opini) dari komentar YouTube secara otomatis.  
Model dilatih menggunakan algoritma **Support Vector Machine (SVM)** dipadukan dengan ekstraksi fitur teks (TF-IDF).
""")

# ==========================================================
# === URL GOOGLE DRIVE UNTUK MODEL & VECTORIZER ============
# ==========================================================
MODEL_URL = "https://drive.google.com/uc?id=1MoPTbJQnFbB7sjG9v2MW9t_Nykrnv9h0"
VECTORIZER_URL = "https://drive.google.com/uc?id=1aI4NI5zsOh9biGdK9YRih90PGkELKGJ6"

# Sesuaikan ekstensinya menjadi .joblib
MODEL_PATH = "model_svm_sentimen.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# ==========================================================
# === FUNGSI LOAD MODEL ====================================
# ==========================================================
@st.cache_resource
def load_model():
    # Mengunduh Model SVM
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model SVM dari Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
    # Mengunduh Vectorizer
    if not os.path.exists(VECTORIZER_PATH):
        with st.spinner("📥 Mengunduh Vectorizer dari Google Drive..."):
            gdown.download(VECTORIZER_URL, VECTORIZER_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

try:
    model, vectorizer = load_model()
    st.success("✅ Model SVM dan Vectorizer berhasil dimuat!")
except Exception as e:
    st.warning("⚠️ Menunggu Anda memasukkan Link Google Drive yang valid di dalam kode (MODEL_URL & VECTORIZER_URL).")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# === TAB NAVIGASI =========================================
# ==========================================================
tab1, tab2, tab3 = st.tabs(["🧾 Input Langsung", "📂 Upload CSV Komentar", "📘 Tentang Aplikasi"])

# ==========================================================
# === TAB 1: INPUT LANGSUNG ================================
# ==========================================================
with tab1:
    st.header("💬 Analisis Komentar Tunggal")
    st.markdown("Masukkan teks komentar YouTube untuk mengetahui apakah sentimennya Positif, Negatif, atau Netral.")

    user_input = st.text_area("Ketik komentar di sini:", height=150, placeholder="Contoh: Videonya sangat bermanfaat dan penjelasannya mudah dipahami bang!")

    if st.button("🔍 Analisis Sentimen", use_container_width=True):
        if user_input.strip() == "":
            st.error("❌ Teks komentar tidak boleh kosong!")
        else:
            with st.spinner('Menganalisis teks...'):
                try:
                    # 1. Cleaning text
                    cleaned_text = clean_text(user_input)
                    
                    # 2. Vectorize text (Ubah teks ke angka)
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    # 3. Prediksi menggunakan SVM
                    pred = model.predict(text_vector)[0]
                    
                    st.subheader("📊 Hasil Prediksi:")
                    
                    # Desain output berdasarkan hasil
                    # Sesuaikan string kondisi di bawah dengan label asli di dataset Anda (misal huruf besar/kecil)
                    pred_str = str(pred).lower()
                    
                    if "positif" in pred_str or pred == 1:
                        st.success(f"Sentimen: **POSITIF** 😊")
                    elif "negatif" in pred_str or pred == 0 or pred == -1:
                        st.error(f"Sentimen: **NEGATIF** 😡")
                    else:
                        st.warning(f"Sentimen: **{str(pred).upper()}** 😐")
                        
                    st.caption(f"Teks setelah dibersihkan (Pre-processed): *{cleaned_text}*")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==========================================================
# === TAB 2: UPLOAD FILE CSV ===============================
# ==========================================================
with tab2:
    st.header("📂 Analisis Sentimen Massal (CSV)")
    st.markdown("Unggah file CSV hasil *scraping* komentar YouTube. Aplikasi akan memprediksi seluruh komentar sekaligus.")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 🧾 Pratinjau Data Awal")
            st.dataframe(df.head())

            # Memilih kolom yang berisi teks komentar
            st.markdown("---")
            text_column = st.selectbox("Pilih kolom yang berisi TEKS KOMENTAR:", df.columns)

            if st.button("🚀 Proses Semua Komentar", use_container_width=True):
                with st.spinner("Sedang memproses dan menganalisis sentimen... Ini mungkin memakan waktu beberapa saat."):
                    # 1. Cleaning & Pre-processing batch
                    df['cleaned_text'] = df[text_column].apply(clean_text)
                    
                    # 2. Hapus baris yang menjadi kosong setelah di-clean
                    df = df[df['cleaned_text'] != ""]
                    
                    # 3. Vectorize batch
                    X_vectorized = vectorizer.transform(df['cleaned_text'])
                    
                    # 4. Predict batch
                    preds = model.predict(X_vectorized)
                    df["Prediksi Sentimen"] = preds

                    st.success("✅ Prediksi massal selesai!")

                    # Tampilkan hasil
                    tampil_cols = [text_column, "cleaned_text", "Prediksi Sentimen"]
                    st.dataframe(df[tampil_cols])

                    # --- Statistik Distribusi ---
                    st.markdown("### 📊 Distribusi Sentimen Komentar")
                    
                    col_chart, col_data = st.columns([2, 1])
                    
                    summary = df["Prediksi Sentimen"].value_counts().reset_index()
                    summary.columns = ["Sentimen", "Jumlah"]
                    summary["Persentase (%)"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(2)
                    
                    with col_data:
                        st.dataframe(summary, use_container_width=True)

                    with col_chart:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        # Warna dinamis
                        colors = ['#4CAF50' if str(x).lower() == 'positif' else '#F44336' if str(x).lower() == 'negatif' else '#FFC107' for x in summary["Sentimen"]]
                        
                        ax.bar(summary["Sentimen"].astype(str), summary["Jumlah"], color=colors)
                        ax.set_xlabel("Kategori Sentimen")
                        ax.set_ylabel("Jumlah Komentar")
                        ax.set_title("Grafik Distribusi Sentimen", pad=10)
                        st.pyplot(fig, use_container_width=False)

                    # --- Unduh hasil ---
                    st.markdown("### 💾 Simpan Hasil")
                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Unduh Hasil Prediksi (CSV)",
                        data=csv_out,
                        file_name="hasil_analisis_sentimen_youtube.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Gagal membaca atau memproses file CSV: {e}")

# ==========================================================
# === TAB 3: TENTANG APLIKASI ==============================
# ==========================================================
with tab3:
    st.write("""
    Aplikasi ini dikembangkan untuk mempermudah analisis sentimen masyarakat terhadap konten YouTube 
    menggunakan pendekatan *Natural Language Processing* (NLP) dan *Machine Learning*.
    """)

    st.subheader("🧠 Framework & Pipeline NLP")
    st.markdown("""
    Sistem ini berjalan mengikuti *pipeline* pemrosesan bahasa alami standar:
    **1. Data Acquisition** — Pengumpulan dataset komentar YouTube (via *scraping* / API).  
    **2. Text Pre-Processing** — Pembersihan teks (menghapus emoji, link, *case folding*).  
    **3. Feature Engineering** — Mengubah teks menjadi representasi vektor numerik menggunakan `TF-IDF Vectorizer`.  
    **4. Modeling** — Klasifikasi menggunakan algoritma `Support Vector Machine (SVM)`.  
    **5. Deployment** — Implementasi interaktif berbasis Streamlit web app.
    """)

    st.subheader("👨‍💻 Pengembang")
    st.markdown("""
    - **Nama:** M. Hibban Ramadhan  
    - **Institusi:** Universitas Lampung  
    - **Teknologi:** Python, Streamlit, Scikit-Learn (SVM), Pandas
    """)
    st.divider()

# ==========================================================
# === FOOTER ===============================================
# ==========================================================
st.markdown("<br><hr><br>", unsafe_allow_html=True)
year = datetime.datetime.now().year

st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 0.9rem; margin-top: 10px;'>
        © {year} <b>M. Hibban Ramadhan</b> — Proyek <i>Analisis Sentimen NLP</i><br>
        Dibangun menggunakan <a href='https://streamlit.io' target='_blank' style='color: #4b9cd3; text-decoration: none;'>Streamlit</a>
    </div>
    """,
    unsafe_allow_html=True
)