import pandas as pd

def get_description(row: pd.Series) -> str:
    desc = row.get("place_description")
    if isinstance(desc, str) and desc.strip():
        return desc
    return str(row.get("gabungan") or "")
