import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ------------------------------- Page Setup ---------------------------------
st.set_page_config(page_title="EcoTourism Neural Recsys", page_icon="🧠", layout="wide")
random.seed(42)

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(os.getcwd())
ART_DIR  = BASE_DIR / "artifacts"

# ------------------------------- Utils --------------------------------------
def format_idr(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return "Rp{:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return str(x)

def get_description(row: pd.Series) -> str:
    desc = row.get("place_description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return str(row.get("gabungan") or "")

def cosine_sim(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    an = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9
    bn = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9
    return (a @ b.T) / (an * bn)

# --------------------------- Artifact Loading --------------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    # --- Items (metadata) ---
    items_csv = ART_DIR / "items.csv"
    if not items_csv.exists():
        st.error("Missing artifacts: `artifacts/items.csv` tidak ditemukan.")
        st.stop()
    items = pd.read_csv(items_csv)

    # Pastikan kolom esensial selalu ada
    must_cols = [
        "place_name", "category", "city", "rating", "price",
        "place_img", "place_map", "place_description", "gabungan", "item_id"
    ]
    for col in must_cols:
        if col not in items.columns:
            if col == "item_id":
                items[col] = items.index.astype(str)
            else:
                items[col] = np.nan
    items["item_id"] = items["item_id"].astype(str)

    # Fallback gabungan bila kosong
    if items["gabungan"].isna().any() or (items["gabungan"].astype(str).str.strip() == "").any():
        def _mk_text(r):
            parts = [
                str(r.get("place_description", "")),
                str(r.get("category", "")),
                str(r.get("city", "")),
                str(r.get("place_name", "")),
            ]
            return " ".join([p for p in parts if isinstance(p, str)]).strip()
        items["gabungan"] = items.apply(_mk_text, axis=1)

    # --- Selaraskan urutan/kolom dengan item_id_map ---
    idmap_path = ART_DIR / "item_id_map.csv"
    if idmap_path.exists():
        idmap = pd.read_csv(idmap_path)
        if "item_id" not in idmap.columns:
            st.error("`item_id_map.csv` harus memiliki kolom 'item_id'.")
            st.stop()
        idmap["item_id"] = idmap["item_id"].astype(str)

        items = idmap.merge(items, on="item_id", how="left", suffixes=("_map", ""))

        # Unify kolom duplikat dari map
        for col in ["place_name", "category", "city"]:
            map_col = f"{col}_map"
            if map_col in items.columns:
                items[col] = items[map_col].combine_first(items.get(col))
                items.drop(columns=[map_col], inplace=True, errors="ignore")

        if "row_idx" in items.columns:
            items = items.sort_values("row_idx").reset_index(drop=True)
        else:
            items.insert(0, "row_idx", np.arange(len(items)))
    else:
        if "row_idx" not in items.columns:
            items.insert(0, "row_idx", np.arange(len(items)))

    # Guard terakhir untuk UI
    for c in ["category", "city", "rating", "price", "gabungan", "place_name", "place_img", "place_map", "place_description"]:
        if c not in items.columns:
            items[c] = np.nan

    items_text = items["gabungan"].astype(str).tolist()

    # --- Load encoder .keras ---
    user_encoder_path = (ART_DIR / "user_encoder.keras")
    item_encoder_path = (ART_DIR / "item_encoder.keras")
    if not user_encoder_path.exists():
        st.error("Missing artifacts: `user_encoder.keras` tidak ditemukan di artifacts/.")
        st.stop()
    if not item_encoder_path.exists():
        st.error("Missing artifacts: `item_encoder.keras` tidak ditemukan di artifacts/.")
        st.stop()

    user_encoder = tf.keras.models.load_model(user_encoder_path.as_posix(), compile=False, safe_mode=False)
    item_encoder = tf.keras.models.load_model(item_encoder_path.as_posix(), compile=False, safe_mode=False)

    # --- Embeddings ---
    def _compute_item_matrix(model, texts, batch=512):
        vecs = []
        for i in range(0, len(texts), batch):
            vecs.append(model(tf.constant(texts[i:i+batch])).numpy())
        return np.vstack(vecs).astype("float32")

    embs_path = ART_DIR / "item_embeddings.npy"
    if embs_path.exists():
        item_embs = np.load(embs_path).astype("float32")
        if item_embs.shape[0] != len(items):
            item_embs = _compute_item_matrix(item_encoder, items_text)
    else:
        item_embs = _compute_item_matrix(item_encoder, items_text)

    return items, items_text, user_encoder, item_encoder, item_embs

items, ITEMS_TEXT, USER_ENCODER, ITEM_ENCODER, ITEM_EMBS = load_artifacts()

# Fast index maps
IDX2ID = items["item_id"].astype(str).tolist()
ID2IDX = {iid: i for i, iid in enumerate(IDX2ID)}

# --------------------------- Retrieval helpers -------------------------------
def liked_centroid(indices):
    if not indices or ITEM_EMBS is None:
        return None
    return ITEM_EMBS[indices].mean(axis=0, keepdims=True)

def filter_mask(categories, cities, max_price):
    m = np.full(len(items), True, dtype=bool)
    if categories:
        cl = [c.lower() for c in categories]
        m &= items["category"].fillna("").apply(lambda s: any(c in str(s).lower() for c in cl)).values
    if cities:
        ct = [c.lower() for c in cities]
        m &= items["city"].fillna("").apply(lambda s: str(s).lower() in ct).values
    if max_price is not None and "price" in items.columns:
        p = items["price"].fillna(np.inf).astype(float).values
        m &= p <= float(max_price)
    return m

def retrieve_ids(query: str, k: int = 50):
    """Neural retrieval: user tower -> dot dengan semua item embeddings (local)."""
    ue = USER_ENCODER(tf.constant([query])).numpy().astype("float32")  # (1, D)
    scores = (ue @ ITEM_EMBS.T).ravel()                                # (N,)
    top_idx = np.argsort(-scores)[:max(k, 1)]
    return top_idx.tolist()

# --------------------------- Session State -----------------------------------
def init_state():
    st.session_state.setdefault("liked_idx", set())
    st.session_state.setdefault("blocked_idx", set())
    st.session_state.setdefault("bookmarked_idx", set())
init_state()

# ------------------------------- Header UI -----------------------------------
st.title("🧠 Neural Recommender — Two-Tower (TFRS)")
st.caption("Search pakai model neural (user tower → item tower). Feed direranking dengan Like/Skip & preferensi kategori.")

st.sidebar.header("Filter")
if "category" not in items.columns:
    items["category"] = ""
if "city" not in items.columns:
    items["city"] = ""

all_categories = sorted(set([
    c.strip()
    for s in items["category"].fillna("").tolist()
    for c in str(s).split(",")
    if c.strip()
]))
sel_cats   = st.sidebar.multiselect("Kategori", options=all_categories)

all_cities = sorted([c for c in items["city"].fillna("").unique() if isinstance(c, str) and c.strip()])
sel_cities = st.sidebar.multiselect("Kota/Kabupaten", options=all_cities)

use_price     = st.sidebar.checkbox("Batasi harga maksimum", value=False)
max_price_val = float(np.nanmax(items["price"].values)) if "price" in items.columns and items["price"].notna().any() else 0.0
price_cap     = st.sidebar.slider(
    "Harga Maksimum (IDR)", 0.0, max_price_val, min(max_price_val, 100_000.0), 1_000.0
) if use_price else None

st.sidebar.header("Feedback")
use_fb = st.sidebar.toggle("Aktifkan reranking Like/Skip", value=True)
alpha  = st.sidebar.slider("Boost ke Like (α)", 0.0, 2.0, 0.6, 0.05, disabled=not use_fb)
beta   = st.sidebar.slider("Penalty Skip (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
gamma  = st.sidebar.slider("Boost kategori Like (γ)", 0.0, 0.2, 0.02, 0.005, disabled=not use_fb)

if st.sidebar.button("Reset Like/Skip", type="secondary"):
    st.session_state.liked_idx.clear()
    st.session_state.blocked_idx.clear()
    st.sidebar.success("Preferensi sesi direset.")
    st.rerun()

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)}")
    with c2: st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)}")
    with c3: st.markdown(f"**Bookmarks (🔖):** {len(st.session_state.bookmarked_idx)}")

# --------------------------------- Tabs --------------------------------------
tab_feed, tab_search, tab_book = st.tabs(["🏠 Feed", "🔎 Search", "🔖 Bookmarks"])

# --------------------------------- FEED --------------------------------------
with tab_feed:
    mask = filter_mask(sel_cats or [], sel_cities or [], price_cap)
    cand = np.where(mask)[0].tolist()
    if not cand:
        st.warning("Tidak ada item untuk filter saat ini.")
    else:
        # Base scores: centroid Like (cosine); fallback rating normalized
        if st.session_state.liked_idx:
            cent = liked_centroid(list(st.session_state.liked_idx))
            base = (cosine_sim(ITEM_EMBS[cand], cent)[:, 0] if cent is not None else np.zeros(len(cand)))
        else:
            r = items.iloc[cand]["rating"].fillna(0.0).astype(float).to_numpy()
            # np.ptp(r)
            rng = float(np.ptp(r))  # r.max() - r.min()
            denom = rng if rng > 1e-9 else 1.0
            base = (r - float(np.min(r))) / denom if len(r) > 0 else np.zeros(len(cand))

        def rerank(indices, base_s):
            liked   = list(st.session_state.liked_idx)
            blocked = set(st.session_state.blocked_idx)
            cat_pref = {}
            if liked:
                cats = items.iloc[liked]["category"].fillna("").apply(lambda s: str(s).split(",")[0].strip())
                for c in cats:
                    if c:
                        cat_pref[c] = cat_pref.get(c, 0) + 1
            scores = {}
            for i, idx in enumerate(indices):
                s = float(base_s[i])
                if idx in blocked:
                    s -= beta
                if liked and gamma > 0:
                    cat = str(items.iloc[idx]["category"]).split(",")[0].strip()
                    if cat:
                        s += gamma * cat_pref.get(cat, 0)
                # alpha * similarity boost (sudah relevan bila base = cosine ke centroid)
                s_final = s + (alpha * base_s[i] if use_fb and liked else 0.0)
                scores[idx] = s_final
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        ranked = rerank(cand, base)
        top_n = st.slider("Jumlah item (Top-N)", 5, 40, 12, 1)

        for ridx, _ in ranked[:top_n]:
            row = items.iloc[int(ridx)]
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    img = row.get("place_img")
                    if isinstance(img, str) and img.startswith(("http://", "https://")):
                        st.image(img, width='stretch')
                with c2:
                    st.subheader(row.get("place_name") or "-")
                    st.markdown(f"**Kategori:** {row.get('category') or '-'}  \n**Kota:** {row.get('city') or '-'}")
                    st.markdown(
                        f"**Rating:** {'-' if pd.isna(row.get('rating')) else round(float(row.get('rating')), 2)}  \n"
                        f"**Harga:** {format_idr(row.get('price'))}"
                    )
                    mp = row.get("place_map")
                    if isinstance(mp, str) and mp.startswith(("http://", "https://")):
                        st.link_button("Buka peta", mp)
                    with st.expander("Lihat deskripsi"):
                        st.write(get_description(row) or "-")
                    b1, b2, b3, _ = st.columns([1, 1, 1, 6])
                    with b1:
                        if st.button("⭐ Suka", key=f"like_feed_{ridx}"):
                            st.session_state.liked_idx.add(int(ridx))
                            st.session_state.blocked_idx.discard(int(ridx))
                            st.rerun()
                    with b2:
                        if st.button("🚫 Skip", key=f"skip_feed_{ridx}"):
                            st.session_state.blocked_idx.add(int(ridx))
                            st.session_state.liked_idx.discard(int(ridx))
                            st.rerun()
                    with b3:
                        label = "Hapus 🔖" if int(ridx) in st.session_state.bookmarked_idx else "🔖 Bookmark"
                        if st.button(label, key=f"bm_feed_{ridx}"):
                            if int(ridx) in st.session_state.bookmarked_idx:
                                st.session_state.bookmarked_idx.remove(int(ridx))
                            else:
                                st.session_state.bookmarked_idx.add(int(ridx))
                            st.rerun()

# -------------------------------- SEARCH ------------------------------------
with tab_search:
    q = st.text_input("Kueri (contoh: pantai aceh snorkeling, gunung camping bandung, dst.)")
    topk = st.slider("Top-K", 10, 100, 30, 5)
    if st.button("Cari", type="primary") and q.strip():
        idxs = retrieve_ids(q.strip(), k=topk * 3)
        if not idxs:
            st.warning("Tidak ada hasil.")
        else:
            m = filter_mask(sel_cats or [], sel_cities or [], price_cap)
            cand = [i for i in idxs if m[i]]
            if not cand:
                st.warning("Tidak ada hasil setelah filter.")
            else:
                # base ranking = urutan dari retrieval (konversi ke skor linier)
                base = np.linspace(1.0, 0.0, num=len(cand), endpoint=False)

                def rerank(indices, base_s):
                    liked   = list(st.session_state.liked_idx)
                    blocked = set(st.session_state.blocked_idx)
                    cat_pref = {}
                    if liked:
                        cats = items.iloc[liked]["category"].fillna("").apply(lambda s: str(s).split(",")[0].strip())
                        for c in cats:
                            if c:
                                cat_pref[c] = cat_pref.get(c, 0) + 1
                    scores = {}
                    for i, idx in enumerate(indices):
                        s = float(base_s[i])
                        if idx in blocked:
                            s -= beta
                        if liked and gamma > 0:
                            cat = str(items.iloc[idx]["category"]).split(",")[0].strip()
                            if cat:
                                s += gamma * cat_pref.get(cat, 0)
                        s_final = s + (alpha * base_s[i] if use_fb and liked else 0.0)
                        scores[idx] = s_final
                    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

                ranked = rerank(cand, base)
                for ridx, _ in ranked[:topk]:
                    row = items.iloc[int(ridx)]
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            img = row.get("place_img")
                            if isinstance(img, str) and img.startswith(("http://", "https://")):
                                st.image(img, width='stretch')
                        with c2:
                            st.subheader(row.get("place_name") or "-")
                            st.markdown(f"**Kategori:** {row.get('category') or '-'}  \n**Kota:** {row.get('city') or '-'}")
                            st.markdown(
                                f"**Rating:** {'-' if pd.isna(row.get('rating')) else round(float(row.get('rating')), 2)}  \n"
                                f"**Harga:** {format_idr(row.get('price'))}"
                            )
                            mp = row.get("place_map")
                            if isinstance(mp, str) and mp.startswith(("http://", "https://")):
                                st.link_button("Buka peta", mp)
                            with st.expander("Lihat deskripsi"):
                                st.write(get_description(row) or "-")
                            b1, b2, b3, _ = st.columns([1, 1, 1, 6])
                            with b1:
                                if st.button("⭐ Suka", key=f"like_search_{ridx}"):
                                    st.session_state.liked_idx.add(int(ridx))
                                    st.session_state.blocked_idx.discard(int(ridx))
                                    st.rerun()
                            with b2:
                                if st.button("🚫 Skip", key=f"skip_search_{ridx}"):
                                    st.session_state.blocked_idx.add(int(ridx))
                                    st.session_state.liked_idx.discard(int(ridx))
                                    st.rerun()
                            with b3:
                                label = "Hapus 🔖" if int(ridx) in st.session_state.bookmarked_idx else "🔖 Bookmark"
                                if st.button(label, key=f"bm_search_{ridx}"):
                                    if int(ridx) in st.session_state.bookmarked_idx:
                                        st.session_state.bookmarked_idx.remove(int(ridx))
                                    else:
                                        st.session_state.bookmarked_idx.add(int(ridx))
                                    st.rerun()
    else:
        st.info("Masukkan kueri untuk mencari.")

# ------------------------------ BOOKMARKS ------------------------------------
with tab_book:
    bms = list(st.session_state.bookmarked_idx)
    if not bms:
        st.info("Belum ada item yang di-bookmark.")
    else:
        st.success(f"{len(bms)} item di-bookmark.")
        for ridx in bms:
            row = items.iloc[int(ridx)]
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                with c1:
                    img = row.get("place_img")
                    if isinstance(img, str) and img.startswith(("http://", "https://")):
                        st.image(img, width='stretch')
                with c2:
                    st.subheader(row.get("place_name") or "-")
                    st.markdown(f"**Kategori:** {row.get('category') or '-'}  \n**Kota:** {row.get('city') or '-'}")
                    st.markdown(
                        f"**Rating:** {'-' if pd.isna(row.get('rating')) else round(float(row.get('rating')), 2)}  \n"
                        f"**Harga:** {format_idr(row.get('price'))}"
                    )
                    mp = row.get("place_map")
                    if isinstance(mp, str) and mp.startswith(("http://", "https://")):
                        st.link_button("Buka peta", mp)
                    with st.expander("Lihat deskripsi"):
                        st.write(get_description(row) or "-")
                    b1, b2, b3, _ = st.columns([1, 1, 1, 6])
                    with b1:
                        if st.button("⭐ Suka", key=f"like_book_{ridx}"):
                            st.session_state.liked_idx.add(int(ridx))
                            st.session_state.blocked_idx.discard(int(ridx))
                            st.rerun()
                    with b2:
                        if st.button("🚫 Skip", key=f"skip_book_{ridx}"):
                            st.session_state.blocked_idx.add(int(ridx))
                            st.session_state.liked_idx.discard(int(ridx))
                            st.rerun()
                    with b3:
                        if st.button("Hapus 🔖", key=f"bm_book_{ridx}"):
                            if int(ridx) in st.session_state.bookmarked_idx:
                                st.session_state.bookmarked_idx.remove(int(ridx))
                            st.rerun()

# -------------------------------- Footer -------------------------------------
st.write("---")
st.caption(
    "Neural Recommender (TFRS two-tower). Artefak: user_encoder.keras, item_encoder.keras, "
    "item_embeddings.npy, item_id_map.csv, items.csv."
)
