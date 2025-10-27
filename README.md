# Indo Ecotourism — Neural Recommender (TensorFlow Recommenders)

Rekomender destinasi ekowisata berbasis **two-tower retrieval** (user tower & item tower) dengan **TensorFlow Recommenders (TFRS)**.  

---

## 🔧 Fitur Utama

- **Two-Tower Text Retrieval:** kedua tower berbagi `TextVectorization` & shared embedding.  
- **EDA Ringkas:** distribusi rating, top-kategori, panjang teks.  
- **Evaluasi Offline:** `recall@{1,5,10,20}` dan `MRR`.  
- **Export Artefak Lengkap:**  
  - `user_encoder.keras` (tower user)  
  - `item_encoder.keras` (tower item)  
  - `item_embeddings.npy` (precompute, `(N, D)`)  
  - `item_id_map.csv` (stabilisasi urutan row ↔ `item_id`)  
  - `items.csv` (input metadata)  
- **Aplikasi Streamlit:**  
  - Tab **Search** (neural retrieval),  
  - Tab **Feed** (cosine ke centroid Like + reranking Like/Skip),  
  - Tab **Bookmarks**.  
  - Filter kategori/kota/harga, panel metrik Like/Skip/Bookmarks.

---

## 🗂 Struktur Proyek

```
.
├── app.py                       # Aplikasi Streamlit (inference)
├── notebook
    ├── indo_ecotourism_neural_recsys.ipynb  # Notebook training + export
    └── indo_ecotourism_neural_recsys.py  # versi py
├── requirements.txt
└── artifacts/
    ├── items.csv                # input metadata
    ├── user_encoder.keras       # disimpan oleh notebook
    ├── item_encoder.keras       # disimpan oleh notebook
    ├── item_embeddings.npy      # disimpan oleh notebook
    └── item_id_map.csv          # disimpan oleh notebook
```

---

## 📦 Dependensi

`requirements.txt`:

```txt
absl-py==2.3.1
altair==5.5.0
astunparse==1.6.3
attrs==25.4.0
blinker==1.9.0
cachetools==6.2.1
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.3.0
flatbuffers==25.9.23
gast==0.6.0
gitdb==4.0.12
GitPython==3.1.45
google-pasta==0.2.0
grpcio==1.76.0
h5py==3.15.1
idna==3.11
Jinja2==3.1.6
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
keras==3.11.3
libclang==18.1.1
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mdurl==0.1.2
ml_dtypes==0.5.3
namex==0.1.0
narwhals==2.9.0
numpy==2.3.4
opt_einsum==3.4.0
optree==0.17.0
packaging==25.0
pandas==2.3.3
pillow==11.3.0
protobuf==6.33.0
pyarrow==22.0.0
pydeck==0.9.1
Pygments==2.19.2
python-dateutil==2.9.0.post0
pytz==2025.2
referencing==0.37.0
requests==2.32.5
rich==14.2.0
rpds-py==0.28.0
setuptools==80.9.0
six==1.17.0
smmap==5.0.2
streamlit==1.50.0
tenacity==9.1.2
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.2.0
toml==0.10.2
tornado==6.5.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
watchdog==6.0.0
Werkzeug==3.1.3
wheel==0.45.1
wrapt==2.0.0

```

---

## 🚀 Quick Start

### 1) Siapkan environment

```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

### 2) Siapkan data

Letakkan `artifacts/items.csv`. Minimal schema:

| kolom              | tipe       | keterangan                                               |
|--------------------|------------|----------------------------------------------------------|
| item_id            | string     | opsional (kalau tidak ada, dibuat dari index)           |
| place_name         | string     | nama destinasi                                           |
| category           | string     | kategori (boleh multi, dipisah koma)                    |
| city               | string     | kota/kabupaten                                           |
| rating             | float      | rating angka (opsional)                                  |
| price              | float      | harga (opsional)                                         |
| place_img          | url        | link gambar (opsional)                                   |
| place_map          | url        | link peta (opsional)                                     |
| place_description  | string     | deskripsi (opsional)                                     |
| gabungan           | string     | teks gabungan (opsional, akan dibuat otomatis jika kosong)|

### 3) Latih model & export artefak

Buka `indo_ecotourism_neural_recsys.ipynb` (Colab/lokal), jalankan semua sel.  
Notebook akan:

- EDA → Pairs (query per item) → Two-Tower → Training → Visualisasi → Evaluasi → Export artefak:
  - `artifacts/user_encoder.keras`
  - `artifacts/item_encoder.keras`
  - `artifacts/item_embeddings.npy`
  - `artifacts/item_id_map.csv`

### 4) Jalankan aplikasi

```bash
streamlit run app.py
```

Buka URL yang ditampilkan (`http://localhost:8501`).

---

## 🧪 Detail Notebook

### [1] Setup & Install
- Menginstal TF + TFRS, set seed (`SEED=42`), siapkan folder `artifacts/`.

### [2] Load Data + EDA
- Membaca `artifacts/items.csv`, melengkapi kolom penting, membentuk `gabungan` jika belum ada.
- Visual: histogram rating, top-10 kategori, panjang teks gabungan.

### [3] Pairing Query–Item
- Membuat beberapa bentuk query per item:
  - **Structured** (kategori, kota, nama),
  - **Template** (“wisata {kategori} di {kota}”),
  - **Keyword** (sampling 6/8/10 token).
- Split **row-wise** 90/10 untuk train/val.

### [4] Two-Tower (Shared Vectorizer)
- `TextVectorization` bersama (vocab gabungan query + item).
- Shared Embedding.
- Tower: `GlobalAveragePooling1D` → Dense → Dropout → Dense → LayerNorm → ReLU → UnitNormalization.
- Dimensi embedding default **128**.

### [5] Training
- `tf.data` untuk train/val.
- Loss: `CategoricalCrossentropy(from_logits=True)` via `tfrs.tasks.Retrieval` (in-batch negatives).
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, **CleanLogger** (log rapi & profesional).

### [6] Visualisasi
- Plot kurva `loss` vs `val_loss`.

### [7] Evaluasi Offline
- Hitung `recall@{1,5,10,20}` dan `MRR` menggunakan dot product U·Vᵀ.

### [8] Export Artefak
- `item_id_map.csv` (map `row_idx` ↔ `item_id`).
- Simpan `user_encoder.keras` & `item_encoder.keras`.
- Precompute `item_embeddings.npy` ukuran `(N_items, EMB_DIM)`.

### [9] Inference Sanity Check
- Fungsi `retrieve_local(query, k)` → top-k `(item_id, score, place_name)`.
- **Tanpa** save/load BruteForce index TFRS (menghindari isu Lambda/safe_mode).

---

## 🖥️ Aplikasi Streamlit (`app.py`)

### Cara Kerja
- Load artefak: `items.csv`, `user_encoder.keras`, `item_encoder.keras`, `item_embeddings.npy`, `item_id_map.csv` (opsional).
- **Search:** `ue = user_encoder(query)`, skor `ue @ item_embs.T`, ambil top-k.
- **Feed:** jika user sudah Like, pusatkan ke centroid Like → cosine similarity; bila belum, fallback normalisasi rating.  
  Reranking mempertimbangkan:
  - **α**: boost kesamaan (Like),
  - **β**: penalti Skip,
  - **γ**: preferensi kategori dari Like.

### Fitur UI
- Filter **Kategori** / **Kota** / **Harga max**.
- Panel jumlah **Liked/Skipped/Bookmarks**.
- Aksi **⭐ Suka**, **🚫 Skip**, **🔖 Bookmark** per item.
- Tab **Bookmarks** untuk daftar simpanan.

---

## ⚙️ Penyesuaian & Tuning

- **Dimensi Embedding:** `EMB_DIM=128` (ubah di notebook).
- **Vocab:** `MAX_TOKENS=40_000` (ubah sesuai dataset).
- **Panjang sequence:** `SEQ_LEN=64` (di versi final tower memakai panjang sama untuk user & item).
- **Hyperparameter Feed:** slider α, β, γ di sidebar app.
- **Recall Target:** atur variasi query generation di notebook (bagian pairs) agar lebih robust.

---

## 🧱 Kompatibilitas & Catatan Teknis

- **Keras 3 + Lambda**: Model disimpan sebagai `.keras`; **load** pakai `safe_mode=False` (sudah ditangani di app).  
- **NumPy 2.0**: method `.ptp()` pada ndarray dihapus → gunakan `np.ptp(arr)`. App sudah diperbaiki.  
- **GPU Warnings**: jika tidak ada CUDA, TF akan jalan di CPU; warning bisa diabaikan.  
- **SavedModel TFRS Index**: sengaja **tidak** digunakan untuk menghindari error deserialisasi Lambda/safe_mode.

---

## 🔁 Reproducibility

- Seed diset ke `SEED=42` untuk `random`, `numpy`, dan `tf.random`.  
- Meski begitu, adanya non-determinism (mis. op paralel) dapat memengaruhi bitwise-identical reproducibility.

---

## 🧱 Batasan

- Retrieval berbasis teks gabungan — kualitas sangat dipengaruhi kualitas `gabungan` & pairs query.  
- Tidak ada model ranking terpisah; reranking sederhana (Like/Skip + kategori).  
- Belum ada personalisasi persistensi lintas sesi (saat ini menggunakan `st.session_state`).

---
