# 🚗 Prediksi Penjualan Mobil Bekas

Perbandingan kinerja algoritma **Naive Bayes**, **Decision Tree**, dan **KNN** dalam klasifikasi potensi kecepatan penjualan mobil bekas.

## 📋 Deskripsi

Website ini menampilkan hasil prediksi dari **3 algoritma machine learning** yang berjalan sepenuhnya di browser (client-side):

| Algoritma | Pendekatan |
|-----------|------------|
| **Naive Bayes** (Gaussian) | Probabilistik — menghitung likelihood berdasarkan distribusi Gaussian |
| **Decision Tree** (Gini Index) | Rule-based — traversal pohon keputusan |
| **KNN** (Euclidean Distance) | Instance-based — mencari K tetangga terdekat |

**Target Klasifikasi:**
- ✅ **Cepat Terjual** — terjual dalam ≤30 hari
- ⏳ **Lama Terjual** — terjual dalam >30 hari

## 🔐 Halaman Login

Website dilengkapi dengan halaman login untuk keamanan akses.

**Akun Demo untuk Pengujian:**

| Field | Nilai |
|-------|-------|
| Username | `demo` |
| Password | `demo` |

> Kredensial demo juga ditampilkan langsung di halaman login agar penguji dapat langsung mencoba.

## 📁 Struktur File

```
used_car_sales_prediction/
├── public/
│   ├── login.html         # Halaman login (username & password)
│   ├── index.html         # Halaman utama (prediksi)
│   ├── style.css          # Tema warm amber/gold automotive
│   ├── app.js             # Implementasi NB, DT, KNN (pure JS)
│   ├── dataset_mobil.csv  # Dataset untuk cascading dropdown
│   └── models/
│       └── model_data.json # Data model (dari notebook)
├── notebook_prediksi_mobil_bekas.ipynb  # Google Colab notebook
├── dataset_mobil.csv                    # Dataset bersih
├── vercel.json                          # Konfigurasi Vercel
└── README.md
```

## 🎨 Desain & Tema

Website menggunakan tema **Warm Automotive** dengan palet warna:

| Warna | Hex | Kegunaan |
|-------|-----|----------|
| 🟠 Amber | `#d97706` | Aksen utama, tombol, badge |
| 🔶 Orange | `#ea580c` | Gradient, aksen sekunder |
| 🟢 Emerald | `#059669` | Aksen hijau, indikator sukses |

Fitur desain:
- **Dark glassmorphism** dengan backdrop blur
- **Animated background blobs** untuk kesan dinamis
- **Micro-animations** pada hover dan transisi
- **Responsive layout** untuk desktop & mobile

## 🔬 Dataset

Sumber: [Kaggle — Used Car Sales](https://www.kaggle.com/datasets/nfathia/used-car-sales) oleh **nfathia**

Dataset terdiri dari 4 tabel relasional:
- `ads.csv` — Data iklan mobil (brand, transmisi, tahun, km, harga)
- `bids.csv` — Data penawaran (bid amount, status, tanggal)
- `buyers.csv` — Data pembeli
- `sellers.csv` — Data penjual

**Fitur yang digunakan:**
- `car_brand` — Merek mobil
- `transmission` — Jenis transmisi (Manual/Automatic)
- `year_of_manufacture` — Tahun pembuatan
- `mileage` — Jarak tempuh (km)
- `price` — Harga jual

## 🚀 Cara Menjalankan

### 1. Jalankan Notebook di Google Colab

1. Buka `notebook_prediksi_mobil_bekas.ipynb` di Google Colab
2. Upload Kaggle API key (`kaggle.json`) — dapatkan dari [kaggle.com/settings](https://www.kaggle.com/settings)
3. Jalankan semua cell dari atas ke bawah
4. Download 2 file yang dihasilkan:
   - `models/model_data.json`
   - `dataset_mobil.csv`

### 2. Letakkan File Output

```bash
# Letakkan model_data.json di:
public/models/model_data.json

# Letakkan dataset_mobil.csv di:
dataset_mobil.csv
```

### 3. Jalankan Lokal

```bash
npx serve public
```

Buka browser di `http://localhost:3000`

> Halaman pertama yang muncul adalah **login.html**. Masukkan username `demo` dan password `demo` untuk mengakses sistem prediksi.

### 4. Deploy ke Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## 🛠 Teknologi

- **Frontend:** HTML, CSS (Dark Glassmorphism), Vanilla JavaScript
- **Tema:** Warm Amber/Gold Automotive (non-AI look)
- **ML Engine:** Pure JavaScript (no TensorFlow.js needed)
- **Autentikasi:** Session-based (sessionStorage)
- **Notebook:** Python, scikit-learn, pandas, matplotlib, seaborn
- **Deployment:** Vercel (static site)
- **Font:** Inter (Google Fonts)
- **Chart:** Chart.js

## 📊 Metodologi

1. **Pengumpulan Data** — Download dari Kaggle
2. **Preprocessing** — Merge 4 tabel, encoding, scaling (StandardScaler)
3. **Pembagian Data** — 80% training, 20% testing
4. **Pelatihan Model** — Naive Bayes, Decision Tree, KNN
5. **Evaluasi** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
6. **Cross-Validation** — 10-Fold Stratified
7. **Export** — Model diekspor ke JSON untuk website

## 📄 Lisensi

Dataset: [MIT License](https://www.mit.edu/~amini/LICENSE.md)
