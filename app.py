import os, random
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st

from eco_recsys.data import load_artifacts
from eco_recsys.state import init_session
from eco_recsys.ui import sidebar_filters, sidebar_knobs, status_chips, card
from eco_recsys.cbf import build_feed, build_search

st.set_page_config(page_title="EcoTourism Neural Recsys", page_icon="🧠", layout="wide")
random.seed(42)

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(os.getcwd())
ART_DIR  = BASE_DIR / "artifacts"

ITEMS, ITEMS_TEXT, USER_ENCODER, ITEM_ENCODER, ITEM_EMBS = load_artifacts(ART_DIR)

init_session()

st.title("🧠 Neural Recommender — Two-Tower (TFRS)")
st.caption("Retrieval pakai user/item encoder. Feed & Search direranking dengan **User Feedback Weighting (UFW)** + **MMR** & **Serendipity**.")

filters = sidebar_filters(ITEMS)
knobs   = sidebar_knobs()
status_chips()

tab_feed, tab_search, tab_book = st.tabs(["🏠 Feed", "🔎 Search", "🔖 Bookmarks"])

with tab_feed:
    ranked = build_feed(ITEMS, ITEM_EMBS, filters, knobs,
                        liked_idx=list(st.session_state.liked_idx),
                        blocked_idx=list(st.session_state.blocked_idx))
    if not ranked:
        st.warning("Tidak ada item untuk filter saat ini. Coba longgarkan filter.")
    else:
        for gid, _ in ranked:
            card(ITEMS.iloc[int(gid)], int(gid), keyprefix="feed")

with tab_search:
    q = st.text_input("Kueri (contoh: pantai aceh snorkeling, gunung camping bandung, savana, kebun teh, dsb.)")
    topk = st.slider("Top-K", 5, 50, 12, 1)
    if st.button("Cari", type="primary") and q.strip():
        ranked = build_search(USER_ENCODER, ITEM_EMBS, ITEMS, q, filters, knobs,
                              liked_idx=list(st.session_state.liked_idx),
                              blocked_idx=list(st.session_state.blocked_idx),
                              topk_request=topk)
        if not ranked:
            st.warning("Tidak ada hasil untuk kueri & filter saat ini.")
        else:
            for gid, _ in ranked:
                card(ITEMS.iloc[int(gid)], int(gid), keyprefix="search")
    else:
        st.info("Masukkan kueri lalu klik **Cari**.")

with tab_book:
    bms = list(st.session_state.bookmarked_idx)
    if not bms:
        st.info("Belum ada item yang di-bookmark.")
    else:
        for gid in bms:
            card(ITEMS.iloc[int(gid)], int(gid), keyprefix="book")

st.write("---")
st.caption("Artifacts: items.csv • user_encoder.keras • item_encoder.keras • item_embeddings.npy • item_id_map.csv (opsional).")
