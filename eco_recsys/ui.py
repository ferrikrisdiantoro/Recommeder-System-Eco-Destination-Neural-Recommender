from typing import Dict
import numpy as np, pandas as pd, streamlit as st
from .utils import format_idr
from .text import get_description
from .state import like, skip, toggle_bookmark

def sidebar_filters(items: pd.DataFrame) -> Dict:
    st.sidebar.header("Filter")
    all_categories = sorted(set([c.strip() for s in items["category"].fillna("").tolist() for c in str(s).split(",") if c.strip()]))
    sel_cats   = st.sidebar.multiselect("Kategori", options=all_categories)
    all_cities = sorted([c for c in items["city"].fillna("").unique() if isinstance(c, str) and c.strip()])
    sel_cities = st.sidebar.multiselect("Kota/Kabupaten", options=all_cities)
    use_price     = st.sidebar.checkbox("Batasi harga maksimum", value=False)
    max_price_val = float(np.nanmax(items["price"].values)) if "price" in items.columns and items["price"].notna().any() else 0.0
    price_cap     = st.sidebar.slider("Harga Maksimum (IDR)", 0.0, max_price_val, min(max_price_val, 100_000.0), 1_000.0) if use_price else None
    return {"categories": sel_cats or [], "cities": sel_cities or [], "max_price": price_cap}

def sidebar_knobs() -> Dict:
    st.sidebar.header("Pengaturan Feed & Rerank")
    top_n_feed   = st.sidebar.slider("Jumlah item Feed (Top-N)", 5, 40, 12, 1)
    mmr_lambda_f = st.sidebar.slider("MMR λ (Feed & Search)", 0.0, 1.0, 0.7, 0.05)
    per_cat_cap  = st.sidebar.slider("Batas per kategori", 0, 6, 2, 1)
    serendip     = st.sidebar.slider("Serendipity (%)", 0, 30, 15, 5)

    st.sidebar.header("User Feedback Weighting (UFW)")
    use_fb = st.sidebar.toggle("Aktifkan reranking Like/Skip", value=True)
    alpha  = st.sidebar.slider("Boost ke Like (α)", 0.0, 2.0, 0.6, 0.05, disabled=not use_fb)
    beta   = st.sidebar.slider("Penalty Skip (β)", 0.0, 2.0, 0.7, 0.05, disabled=not use_fb)
    gamma  = st.sidebar.slider("Boost kategori Like (γ)", 0.0, 0.2, 0.02, 0.005, disabled=not use_fb)
    if st.sidebar.button("Reset Like/Skip", type="secondary"):
        st.session_state.liked_idx.clear()
        st.session_state.blocked_idx.clear()
        st.sidebar.success("Preferensi sesi direset.")
        st.rerun()
    return {"top_n": top_n_feed, "mmr_lambda": mmr_lambda_f, "per_cat_cap": per_cat_cap, "serendip": serendip,
            "use_fb": use_fb, "alpha": alpha, "beta": beta, "gamma": gamma}

def status_chips():
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"**Liked (⭐):** {len(st.session_state.liked_idx)} item")
        with c2: st.markdown(f"**Skipped (🚫):** {len(st.session_state.blocked_idx)} item")
        with c3: st.markdown(f"**Bookmarks (🔖):** {len(st.session_state.bookmarked_idx)} item")

def card(row: pd.Series, gid: int, keyprefix: str):
    with st.container(border=True):
        cols = st.columns([1,3])
        with cols[0]:
            img = row.get("place_img")
            if isinstance(img, str) and img.startswith(("http://","https://")):
                st.image(img, use_container_width=True)
        with cols[1]:
            st.subheader(row.get("place_name") or "-")
            st.markdown(f"**Kategori:** {row.get('category') or '-'}  \n**Kota:** {row.get('city') or '-'}")
            st.markdown(
                f"**Rating:** {'-' if pd.isna(row.get('rating')) else round(float(row.get('rating')), 2)}  \n"
                f"**Harga:** {format_idr(row.get('price'))}"
            )
            mp = row.get("place_map")
            if isinstance(mp, str) and mp.startswith(("http://","https://")):
                st.link_button("Buka peta", mp, width='content')
            with st.expander("Lihat deskripsi"):
                st.write(get_description(row) or "-")
            b1, b2, b3, _ = st.columns([1,1,1,6])
            with b1:
                if st.button("⭐ Suka", key=f"{keyprefix}_like_{gid}"):
                    like(gid); st.rerun()
            with b2:
                if st.button("🚫 Skip", key=f"{keyprefix}_skip_{gid}"):
                    skip(gid); st.rerun()
            with b3:
                label = "Hapus 🔖" if int(gid) in st.session_state.bookmarked_idx else "🔖 Bookmark"
                if st.button(label, key=f"{keyprefix}_bm_{gid}"):
                    toggle_bookmark(gid); st.rerun()
