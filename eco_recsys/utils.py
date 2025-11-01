import math, re
from typing import Iterable, Dict, Any, List
import numpy as np
import pandas as pd

def format_idr(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or math.isinf(x))):
        return "-"
    try:
        return "Rp{:,.0f}".format(float(x)).replace(",", ".")
    except Exception:
        return str(x)

def normalize_minmax(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)

def cosine_sim(A, B):
    A = np.asarray(A, dtype="float32")
    B = np.asarray(B, dtype="float32")
    An = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    Bn = np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    return (A @ B.T) / (An @ Bn.T)

def filter_mask(items: pd.DataFrame, categories: List[str], cities: List[str], max_price):
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
