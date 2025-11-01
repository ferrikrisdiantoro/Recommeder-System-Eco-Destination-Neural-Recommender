from typing import Dict, List, Tuple
import numpy as np
from .utils import cosine_sim, normalize_minmax

def apply_user_feedback_weighting(
    indices: List[int],
    base_scores: np.ndarray,
    items,
    item_embs: np.ndarray,
    liked_idx: List[int],
    blocked_idx: List[int],
    alpha: float = 0.6,
    beta: float = 0.7,
    gamma: float = 0.02
) -> List[Tuple[int, float]]:
    base_map = {int(i): float(base_scores[k]) for k,i in enumerate(indices)}

    cat_pref: Dict[str,int] = {}
    if liked_idx:
        liked_cats = (items.iloc[liked_idx]["category"].fillna("")
                      .apply(lambda s: str(s).split(",")[0].strip()))
        for c in liked_cats:
            if c:
                cat_pref[c] = cat_pref.get(c, 0) + 1

    if liked_idx:
        cent = item_embs[liked_idx].mean(axis=0, keepdims=True)
        like_sims = cosine_sim(item_embs[indices], cent)[:,0]
        like_sims = normalize_minmax(like_sims)
    else:
        like_sims = np.zeros(len(indices), dtype=float)

    blocked = set(map(int, blocked_idx))
    out = []
    for j, gid in enumerate(indices):
        s = base_map.get(int(gid), 0.0)
        if liked_idx:
            s += alpha * like_sims[j]
        if int(gid) in blocked:
            s -= beta
        if cat_pref:
            cat = str(items.iloc[int(gid)]["category"]).split(",")[0].strip()
            if cat:
                s += gamma * cat_pref.get(cat, 0)
        out.append((int(gid), float(s)))
    return sorted(out, key=lambda x: x[1], reverse=True)
