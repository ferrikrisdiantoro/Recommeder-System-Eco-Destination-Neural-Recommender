from typing import List, Tuple, Dict
import numpy as np, tensorflow as tf
from .utils import cosine_sim, normalize_minmax, filter_mask
from .ufw import apply_user_feedback_weighting

# -- Retrieval (Two-tower TFRS): user encoder -> dot with item_embs --
def neural_retrieve(user_encoder: tf.keras.Model, item_embs: np.ndarray, query: str, topk: int = 100) -> Tuple[list, np.ndarray]:
    ue = user_encoder(tf.constant([str(query)])).numpy().astype("float32")
    scores = (ue @ item_embs.T).ravel()
    order  = np.argsort(-scores)[:max(1, topk)]
    return order.tolist(), scores

# -- Base score for Feed (centroid Like or rating) --
def feed_base_scores(items, item_embs: np.ndarray, candidate_idx, liked_idx: List[int]) -> np.ndarray:
    if liked_idx:
        cent = item_embs[liked_idx].mean(axis=0, keepdims=True)
        sims = cosine_sim(item_embs[candidate_idx], cent)[:,0]
        return normalize_minmax(sims)
    else:
        r = items.iloc[candidate_idx]["rating"].fillna(0.0).astype(float).to_numpy()
        return normalize_minmax(r)

# -- MMR diversification --
def mmr_select(item_embs: np.ndarray, idx_all: np.ndarray, base_scores: np.ndarray,
               top_n: int = 20, lambda_mmr: float = 0.7, per_category_cap: int = 0, items=None) -> List[int]:
    idx_all = np.asarray(idx_all, dtype=int)
    base_scores = np.asarray(base_scores, dtype=float)
    if idx_all.size == 0:
        return []
    selected_loc: List[int] = []
    candidates = list(range(len(idx_all)))
    cat_count: Dict[str,int] = {}
    while candidates and len(selected_loc) < min(top_n*3, len(idx_all)):
        best_loc, best_val = None, -1e18
        for loc in candidates:
            gid = int(idx_all[loc])
            if per_category_cap and items is not None:
                cat = str(items.iloc[gid]["category"]).split(",")[0].strip()
                if cat and cat_count.get(cat, 0) >= per_category_cap:
                    continue
            score = float(base_scores[loc])
            if not selected_loc:
                mmr_val = score
            else:
                sel_g = idx_all[selected_loc]
                sims = cosine_sim(item_embs[[gid]], item_embs[sel_g])[0]
                mmr_val = lambda_mmr * score - (1.0 - lambda_mmr) * float(np.max(sims))
            if mmr_val > best_val:
                best_val, best_loc = mmr_val, loc
        if best_loc is None:
            break
        selected_loc.append(best_loc)
        if per_category_cap and items is not None:
            cat = str(items.iloc[int(idx_all[best_loc])]["category"]).split(",")[0].strip()
            if cat:
                cat_count[cat] = cat_count.get(cat, 0) + 1
        candidates.remove(best_loc)
        if len(selected_loc) >= top_n:
            break
    return [int(idx_all[loc]) for loc in selected_loc]

# -- FEED builder --
def build_feed(items, item_embs, filters: Dict, knobs: Dict, liked_idx: List[int], blocked_idx: List[int]):
    mask = filter_mask(items, filters["categories"], filters["cities"], filters["max_price"])
    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return []
    base = feed_base_scores(items, item_embs, idx_all, liked_idx)
    gids = mmr_select(item_embs, idx_all, base, top_n=knobs["top_n"],
                      lambda_mmr=knobs["mmr_lambda"], per_category_cap=knobs["per_cat_cap"], items=items)
    # Serendipity: sisipkan top-rated lain yg belum terambil
    selected = set(gids)
    ser_k = max(0, min(max(1, knobs["top_n"] // 5), int(len(idx_all) * knobs["serendip"] / 100)))
    if ser_k > 0:
        pool = [g for g in idx_all if g not in selected]
        if pool:
            top_pop = items.iloc[pool].copy()
            pool_sorted = list(top_pop.sort_values(["rating"], ascending=False).index)
            import random; random.shuffle(pool_sorted)
            gids.extend(pool_sorted[:ser_k])
    base_for_gids = np.array([base[list(idx_all).index(g)] if g in idx_all else 0.0 for g in gids], dtype=float)
    if knobs["use_fb"]:
        ranked = apply_user_feedback_weighting(
            indices=gids, base_scores=base_for_gids, items=items, item_embs=item_embs,
            liked_idx=liked_idx, blocked_idx=blocked_idx,
            alpha=knobs["alpha"], beta=knobs["beta"], gamma=knobs["gamma"]
        )
    else:
        ranked = sorted([(int(g), float(base_for_gids[i])) for i,g in enumerate(gids)], key=lambda x: x[1], reverse=True)
    return ranked[:knobs["top_n"]]

# -- SEARCH builder --
def build_search(user_encoder, item_embs, items, query: str, filters: Dict, knobs: Dict,
                 liked_idx: List[int], blocked_idx: List[int], topk_request: int = 30):
    if not str(query).strip(): return []
    idxs, _ = neural_retrieve(user_encoder, item_embs, query.strip(), topk=topk_request*4)
    if not len(idxs): return []
    mask = filter_mask(items, filters["categories"], filters["cities"], filters["max_price"])
    cand = [i for i in idxs if mask[i]]
    if not cand: return []
    base = np.linspace(1.0, 0.0, num=len(cand), endpoint=False)
    gids = mmr_select(item_embs, np.array(cand, dtype=int), base, top_n=min(topk_request, len(cand)),
                      lambda_mmr=knobs["mmr_lambda"], per_category_cap=knobs["per_cat_cap"], items=items)
    base_for_gids = np.array([base[cand.index(g)] if g in cand else 0.0 for g in gids], dtype=float)
    if knobs["use_fb"]:
        ranked = apply_user_feedback_weighting(
            indices=gids, base_scores=base_for_gids, items=items, item_embs=item_embs,
            liked_idx=liked_idx, blocked_idx=blocked_idx,
            alpha=knobs["alpha"], beta=knobs["beta"], gamma=knobs["gamma"]
        )
    else:
        ranked = sorted([(int(g), float(base_for_gids[i])) for i,g in enumerate(gids)], key=lambda x: x[1], reverse=True)
    return ranked[:topk_request]
