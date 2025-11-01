from pathlib import Path
from typing import Tuple
import numpy as np, pandas as pd, streamlit as st, tensorflow as tf

@st.cache_resource(show_spinner=True)
def load_artifacts(art_dir: Path) -> Tuple[pd.DataFrame, list, tf.keras.Model, tf.keras.Model, np.ndarray]:
    items_csv = art_dir / "items.csv"
    if not items_csv.exists():
        st.error("Missing artifacts: `artifacts/items.csv` tidak ditemukan.")
        st.stop()
    items = pd.read_csv(items_csv)

    must_cols = ["place_name","category","city","rating","price","place_img","place_map","place_description","gabungan","item_id"]
    for col in must_cols:
        if col not in items.columns:
            if col == "item_id":
                items[col] = items.index.astype(str)
            else:
                items[col] = np.nan
    items["item_id"] = items["item_id"].astype(str)

    if "gabungan" not in items.columns or items["gabungan"].isna().any() or (items["gabungan"].astype(str).str.strip()=="").any():
        def _mk_text(r):
            parts = [str(r.get("place_description","")), str(r.get("category","")), str(r.get("city","")), str(r.get("place_name",""))]
            return " ".join([p for p in parts if isinstance(p,str)]).strip()
        items["gabungan"] = items.apply(_mk_text, axis=1)

    idmap_path = art_dir / "item_id_map.csv"
    if idmap_path.exists():
        idmap = pd.read_csv(idmap_path)
        if "item_id" in idmap.columns:
            idmap["item_id"] = idmap["item_id"].astype(str)
            items = idmap.merge(items, on="item_id", how="left", suffixes=("_map",""))
            for col in ["place_name","category","city"]:
                mc = f"{col}_map"
                if mc in items.columns:
                    items[col] = items[mc].combine_first(items.get(col))
                    items.drop(columns=[mc], inplace=True, errors="ignore")
            if "row_idx" in items.columns:
                items = items.sort_values("row_idx").reset_index(drop=True)
        if "row_idx" not in items.columns:
            items.insert(0, "row_idx", range(len(items)))
    else:
        if "row_idx" not in items.columns:
            items.insert(0, "row_idx", range(len(items)))

    u_path = art_dir / "user_encoder.keras"
    i_path = art_dir / "item_encoder.keras"
    if not u_path.exists() or not i_path.exists():
        st.error("Missing encoder: `user_encoder.keras` & `item_encoder.keras` harus ada di artifacts/.")
        st.stop()
    user_encoder = tf.keras.models.load_model(u_path.as_posix(), compile=False, safe_mode=False)
    item_encoder = tf.keras.models.load_model(i_path.as_posix(), compile=False, safe_mode=False)

    texts = items["gabungan"].astype(str).tolist()

    def _compute_item_matrix(model, texts, batch=512):
        vecs = []
        for i in range(0, len(texts), batch):
            vecs.append(model(tf.constant(texts[i:i+batch])).numpy())
        return np.vstack(vecs).astype("float32")

    embs_path = art_dir / "item_embeddings.npy"
    if embs_path.exists():
        item_embs = np.load(embs_path).astype("float32")
        if item_embs.shape[0] != len(items):
            item_embs = _compute_item_matrix(item_encoder, texts)
    else:
        item_embs = _compute_item_matrix(item_encoder, texts)

    return items, texts, user_encoder, item_encoder, item_embs
