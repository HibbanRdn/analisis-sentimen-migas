# 🧠 Analisis Sentimen Komentar YouTube Menggunakan Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![TF-IDF](https://img.shields.io/badge/Feature_Extraction-TF--IDF-green)
![Model Accuracy](https://img.shields.io/badge/Accuracy-87%25-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📋 Deskripsi Proyek
Proyek ini merupakan implementasi **analisis sentimen komentar YouTube** menggunakan algoritma **Support Vector Machine (SVM)**.  
Tujuannya adalah untuk **mengklasifikasikan opini publik** menjadi tiga kategori sentimen:  
➡️ **Positif**, ⚪ **Netral**, dan ❌ **Negatif**.

Dataset dikumpulkan secara otomatis menggunakan **YouTube Data API v3**, lalu melalui tahapan:
- Pembersihan teks (preprocessing)
- Ekstraksi fitur dengan **TF-IDF**
- Pelatihan model menggunakan **SVM (kernel linear)**
- Evaluasi kinerja model dengan metrik akurasi, presisi, recall, dan F1-score

---

## 🧩 Fitur Utama
✅ Crawling komentar otomatis dari YouTube  
✅ Preprocessing teks bahasa Indonesia (case folding, stopword removal, stemming, tokenizing)  
✅ Ekstraksi fitur TF-IDF  
✅ Klasifikasi sentimen menggunakan SVM  
✅ Evaluasi performa model  
✅ Prediksi sentimen untuk komentar baru  

---

## 📂 Struktur Folder
📦 Analisis-Sentimen-SVM  
├── 📁 dataset/  
│   ├── komentar_youtube.csv  
│   └── komentar_clean.csv  
├── 📁 models/  
│   ├── svm_sentimen.joblib  
│   └── tfidf_vectorizer.joblib  
├── 📁 notebook/  
│   └── analisis_sentimen_svm.ipynb  
├── app.py                     # Aplikasi prediksi komentar baru  
├── requirements.txt           # Daftar dependensi Python  
├── laporan.docx               # Laporan penelitian lengkap  
└── README.md                  # Dokumentasi proyek

---

## ⚙️ Instalasi dan Penggunaan

### 🔧 1. Persiapkan lingkungan Python
Pastikan Python ≥ 3.9 sudah terinstal, kemudian jalankan:
```
pip install -r requirements.txt
```

📒 2. Jalankan notebook

jupyter notebook notebook/analisis_sentimen_svm.ipynb

💬 3. Prediksi komentar baru
```
from joblib import load

# Load model dan vectorizer
model = load('models/svm_sentimen.joblib')
vectorizer = load('models/tfidf_vectorizer.joblib')

# Contoh prediksi
komentar = ["Kasus ini sangat memalukan bagi bangsa"]
komentar_tfidf = vectorizer.transform(komentar)
prediksi = model.predict(komentar_tfidf)
print(prediksi)
```

---

💡 Saran Pengembangan
- Integrasi dengan Streamlit Dashboard untuk analisis sentimen interaktif.
- Eksperimen dengan LSTM atau BERT IndoBERT untuk perbandingan hasil.
- Pengumpulan dataset yang lebih besar dan beragam untuk memperkuat generalisasi model.

---

🧾 Referensi
- Ma’rufudin, & Yudhistira, A. (2025). Analisis sentimen petani milenial pada media sosial X menggunakan algoritma Support Vector Machine (SVM). Jurnal Pendidikan dan Teknologi Indonesia, 5(3).
- Setiawan, H., Utami, E., & Sudarmawan, S. (2021). Analisis sentimen Twitter kuliah online pasca Covid-19 menggunakan algoritma Support Vector Machine dan Naïve Bayes. Jurnal Komtika, 5(1), 43–51.
- Wardhani, D., dkk. (2024). Pengantar Data Mining. Mifandi Mandiri Digital.
- Saputra, J., Maryani, L., Rahmaddeni, D. Wulandari, & Wisnu, E. (n.d.). Analisis performa Naive Bayes dan SVM terhadap sentimen teks media sosial dengan Word2Vec dan SMOTE. Jurnal INSTEK, 10(1).
- Seno, B. A., Widodo, & Prasetya Adhi, B. (n.d.). Penerapan algoritma Support Vector Machine untuk mendeteksi emosi dari teks bahasa Indonesia. PINTER: Jurnal Pendidikan Teknik Informatika dan Komputer, 8(1).
- Supian, A., Revaldo, B. T., Marhadi, N., Rahmaddeni, & Efrizoni, L. (n.d.). Penerapan SVM dan Word2Vec untuk analisis sentimen ulasan pengguna aplikasi DANA. Jurnal Ilmiah Komputasi, 23(3).

---

Pengembang
- Nama: Muhamad Hibban Ramadhan
- NIM: 2315061094
- Program Studi: Teknik Informatika – Konsentrasi Kecerdasan Buatan
- Universitas: Universitas Lampung
- Tahun: 2025
