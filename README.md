# EcoTourism — Neural Recommender (Two‑Tower, TFRS) — **eco_recsys (modular)**

Rekomender destinasi ekowisata berbasis **two‑tower retrieval** (user tower & item tower) dengan **TensorFlow Recommenders** untuk embedding teks, lalu **reranking** menggunakan **User Feedback Weighting (UFW)** + **MMR** + **Serendipity**.

---

## 🔧 Fitur Utama
- **Two‑Tower Retrieval**: user encoder → *dot product* dengan matriks embedding item lokal (`item_embeddings.npy`).
- **UFW Reranking**: boost kemiripan ke **centroid Like** (α), penalti Skip (β), dan preferensi kategori dari Like (γ).
- **Diversifikasi (MMR)**: kontrol keragaman dengan λ dan **batas per‑kategori**.
- **Serendipity**: sisipkan sebagian item populer di luar kandidat utama.
- **UI Streamlit**: tab **Feed**, **Search**, dan **Bookmarks** dengan filter kategori/kota/harga & kontrol α/β/γ.

---

## 🗂 Struktur Proyek
```
eco_recsys_app/
├─ app.py                      # Orkestrator Streamlit (layout + panggil builder)
├─ artifacts/                  # TARUH artefak di sini (lihat bagian "Artefak")
├─ eco_recsys/                 # Paket modular (LOGIKA dipisah per tanggung jawab)
│  ├─ __init__.py
│  ├─ cbf.py                   # retrieval (two‑tower), MMR, builder feed/search
│  ├─ data.py                  # loader/cacher artefak
│  ├─ state.py                 # session state: like/skip/bookmark, reset
│  ├─ text.py                  # util deskripsi
│  ├─ ufw.py                   # User Feedback Weighting (α/β/γ)
│  ├─ ui.py                    # komponen UI (sidebar, kartu item)
│  └─ utils.py                 # format_idr, cosine, min‑max, filter mask
└─ requirements.txt
```

---

## 📦 Artefak yang Diperlukan
Letakkan berkas berikut di folder `artifacts/`:
- `items.csv` — metadata item (wajib). Kolom yang dipakai: `place_name`, `category`, `city`, `rating`, `price`, `place_img`, `place_map`, `place_description`, `gabungan`, `item_id` (opsional). Bila `gabungan` kosong, aplikasi akan membangunnya dari *description+category+city+name*.
- `user_encoder.keras` — tower user (hasil notebook pelatihan).
- `item_encoder.keras` — tower item (hasil notebook pelatihan).
- `item_embeddings.npy` — **opsional**; jika tidak ada, aplikasi akan **menghitung** embedding item dari `item_encoder` saat start.
- `item_id_map.csv` — **opsional**; menstabilkan pemetaan `row_idx ↔ item_id` (berguna bila urutan item penting).

---

## 🚀 Cara Menjalankan
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Jalankan Streamlit
streamlit run app.py
```
Buka URL yang ditampilkan (default: `http://localhost:8501`).

---

## 🧠 Alur Peringkat (sesuai kode)
### 1) Feed (tanpa kueri)
1. **Filter** (kategori/kota/harga) → himpunan kandidat `idx_all`.
2. **Base score**:
   - Jika sudah ada ⭐ **Like**: cosine ke **centroid** embedding item yang disukai (dinormalisasi min‑max).
   - Jika belum ada Like: normalisasi **rating** (proxy popularitas).
3. **MMR** (λ) untuk variasi (opsional batas per‑kategori).
4. **Serendipity**: sisipkan sebagian item populer di luar kandidat MMR.
5. **UFW**: skor akhir = `base + α·sim_ke_centroid_like − β·skip + γ·preferensi_kategori_like`.
6. Render **Top‑N** kartu.

### 2) Search
1. `ue = user_encoder(query)` → `scores = ue @ ITEM_EMBS.T` → ambil **Top‑K** kandidat.
2. Terapkan **filter** → MMR (λ) → **UFW** → tampilkan hasil.

### 3) Bookmarks
- Hanya menampilkan item yang dibookmark; tidak memengaruhi peringkat.

---

## 🎛️ Parameter Penting (Sidebar)
- **Feed**: `Top‑N`, `MMR λ`, `Batas per kategori`, `Serendipity %`.
- **UFW**: `α` (boost Like), `β` (penalti Skip), `γ` (preferensi kategori dari Like).
- **Search**: `Top‑K` hasil.

**State Sesi** (disimpan di `st.session_state`):
- `liked_idx: set[int]`, `blocked_idx: set[int]`, `bookmarked_idx: set[int]`.
- Semua aksi tombol memanggil `st.rerun()` agar urutan langsung terbarui.

---

## 🧪 Evaluasi Offline (Ringkas)
Evaluasi metrik dilakukan di **notebook pelatihan** (bukan di aplikasi):
- Retrieval two‑tower: `recall@{1,5,10,20}`, **MRR** (*dot product* U·Vᵀ).
- Bandingan baseline vs model neural dapat ditambahkan di notebook sesuai kebutuhan.

---

## 🪛 Troubleshooting
- **“Missing artifacts”** → pastikan berkas ada di `artifacts/` sesuai daftar di atas.
- **`item_embeddings.npy` bentuknya tidak cocok** → hapus berkas tersebut agar aplikasi **recompute** embedding item dari `item_encoder`.
- **Keras `safe_mode`** → loader sudah memakai `safe_mode=False` di `data.py`.
- **Lambat saat start** → tanpa `item_embeddings.npy`, aplikasi akan menghitung embedding seluruh item di awal.

---

## 🔌 Opsi Ekstensi
- Ganti *dot product* brute‑force dengan **FAISS/ANN** untuk jutaan item.
- Persistensi feedback ke DB (bukan hanya session) + personalisasi lintas sesi.
- Logging/telemetri (latensi, klik, CTR) dan A/B testing parameter α/β/γ/λ.
- Knowledge snippets/FAQ per kategori untuk tampilan deskripsi yang lebih informatif.

---
