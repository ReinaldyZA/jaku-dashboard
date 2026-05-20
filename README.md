# 🌬️ JakU — Jakarta Kualitas Udara

**Dashboard Klasifikasi Kualitas Udara DKI Jakarta**

Sistem dashboard interaktif berbasis web untuk klasifikasi kualitas udara di DKI Jakarta menggunakan algoritma *machine learning* (Random Forest, XGBoost, dan SVM) berdasarkan data Indeks Standar Pencemar Udara (ISPU).

🔗 **Live Demo:** [jaku-dashboard-2sjagbejjkqhhprykcr5vg.streamlit.app](https://jaku-dashboard-2sjagbejjkqhhprykcr5vg.streamlit.app/)

---

## 📋 Tentang Proyek

JakU dikembangkan sebagai bagian dari tugas akhir (skripsi) Program Studi Teknik Informatika, Universitas Bina Nusantara (BINUS). Proyek ini bertujuan untuk:

1. Membangun model klasifikasi kualitas udara di DKI Jakarta berdasarkan enam parameter polutan (PM10, PM2.5, SO₂, CO, O₃, NO₂) ke dalam tiga kategori ISPU: **BAIK**, **SEDANG**, dan **TIDAK SEHAT**.
2. Membandingkan performa tiga algoritma *machine learning* — Random Forest, XGBoost, dan SVM — menggunakan *hyperparameter tuning* (GridSearchCV) dan *Stratified 5-Fold Cross-Validation*.
3. Menyajikan hasil analisis dalam bentuk dashboard interaktif yang menerapkan prinsip *User-Centered Design* (UCD) berdasarkan *Nielsen's Usability Heuristics*.

Metodologi penelitian mengikuti kerangka kerja **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), dengan pengembangan sistem menggunakan metode **RAD** (*Rapid Application Development*) dan evaluasi usabilitas melalui **SUS** (*System Usability Scale*).

---

## ✨ Fitur Dashboard

Dashboard JakU terdiri dari enam halaman utama:

### 🏠 Beranda
Ringkasan keseluruhan hasil analisis: KPI cards (total data, data final, model terbaik, akurasi), grafik perbandingan performa model, distribusi kategori ISPU, dan alur metodologi CRISP-DM.

### 🔍 Eksplorasi Data
Implementasi fase *Data Understanding* dengan enam tab interaktif: statistik deskriptif, distribusi fitur (histogram), boxplot, heatmap korelasi, missing values, dan distribusi per stasiun pemantauan (peta interaktif + stacked bar chart).

### 🧹 Persiapan Data
Implementasi fase *Data Preparation* dengan empat tab: data cleaning, imputasi median, outlier removal (IQR), dan encoding & split (Label Encoding, stratified 80:20, StandardScaler untuk SVM).

### 🤖 Pemodelan & Evaluasi
Implementasi fase *Modeling* dan *Evaluation* dengan enam tab: best hyperparameters, confusion matrix, classification report, 5-fold cross-validation, feature importance, dan perbandingan performa (tabel + radar chart).

### 🎯 Prediksi Interaktif
Fitur utama klasifikasi *real-time*: pilih model, input enam parameter polutan via slider atau preset skenario (Udara Bersih/Sedang/Buruk), lihat hasil prediksi + tingkat keyakinan + rekomendasi aktivitas + radar chart profil polutan.

### ℹ️ Tentang
Informasi penelitian: tujuan, dataset, metodologi, prinsip UCD, dan kategori ISPU.

---

## 🛠️ Teknologi

| Komponen | Teknologi |
|---|---|
| Framework | Streamlit (Python) |
| Frontend | Streamlit components + custom CSS, Google Fonts (Plus Jakarta Sans) |
| Visualisasi | Plotly Express, Plotly Graph Objects |
| Machine Learning | scikit-learn, XGBoost |
| Pemrosesan Data | pandas, NumPy |
| Deployment | Streamlit Community Cloud |

---

## 📊 Dataset

- **Sumber:** Jakarta Open Data ([ https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=data-indeks-standar-pencemar-udara-ispu-di-provinsi-dki-jakarta&data_no=1 ])
- **Format:** CSV (semicolon-delimited)
- **Jumlah Data Awal:** 3.350 baris
- **Jumlah Data Final:** 3.057 baris (setelah cleaning dan outlier removal)
- **Stasiun Pemantauan:** 5 stasiun (DKI1 Bunderan HI, DKI2 Kelapa Gading, DKI3 Jagakarsa, DKI4 Lubang Buaya, DKI5 Kebon Jeruk)
- **Fitur:** PM10, PM2.5, SO₂, CO, O₃, NO₂
- **Target:** Kategori ISPU (BAIK, SEDANG, TIDAK SEHAT)

---

## 🏆 Hasil

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean ± Std |
|---|---|---|---|---|---|
| Random Forest | 0.9689 | 0.9691 | 0.9689 | 0.9688 | 0.9654 ± 0.0081 |
| **XGBoost** | **0.9771** | **0.9773** | **0.9771** | **0.9770** | **0.9706 ± 0.0082** |
| SVM | 0.9566 | 0.9568 | 0.9566 | 0.9564 | 0.9549 ± 0.0101 |

**Model terbaik: XGBoost** dengan akurasi 97,71% pada data uji.

---

## 🚀 Cara Menjalankan Secara Lokal

### Prasyarat
- Python 3.8 atau lebih baru
- pip (Python package manager)

### Langkah

1. **Clone repositori**
   ```bash
   git clone https://github.com/ReinaldyZA/jaku-dashboard.git
   cd jaku-dashboard
   ```

2. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

4. **Buka browser** di `http://localhost:8501`

---

## 📁 Struktur Direktori

```
jaku-dashboard/
├── app.py               # Kode sumber utama (single-file architecture)
├── Data_ISPU.csv        # Dataset ISPU DKI Jakarta
├── requirements.txt     # Daftar dependensi Python
└── README.md            # Dokumentasi proyek
```

---

## 👥 Tim Pengembang

Proyek ini dikembangkan sebagai tugas akhir (skripsi) oleh mahasiswa Program Studi Teknik Informatika, Universitas Bina Nusantara (BINUS):

| Nama | NIM |
|---|---|
| James Randolph Candra Kusuma | 2602092825 |
| Reinaldy Zulfananda Arkaan | 2602168740 |
| Syafiq Ammar Muhadzdzib | 2602172946 |

**Dosen Pembimbing:** Rezki Yunanda, S.Kom, M.Kom

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan akademis sebagai bagian dari skripsi di Universitas Bina Nusantara (BINUS).

---

<p align="center">
  Dibuat dengan ❤️ menggunakan <a href="https://streamlit.io">Streamlit</a>
</p>
