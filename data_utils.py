import ast
from typing import List
import pandas as pd

from config import EXCHANGE_RATES


def convert_to_usd(value, country):
    """Convert local currency to USD using EXCHANGE_RATES"""
    if pd.isna(value) or pd.isna(country):
        return value
    return value / EXCHANGE_RATES.get(country, 1.0)


def parse_emtec_list(txt: str) -> List[str]:
    """Safely parse the EMTEC list string"""
    try:
        return [] if pd.isna(txt) or txt in ('[]', '') else ast.literal_eval(txt)
    except Exception:
        return []


def create_multiple_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate one row per every EMSEC × EMTEC combination (cross‑product)."""
    expanded_rows: List[pd.Series] = []

    for _, row in df.iterrows():
        # ── 1) Collect valid EMSEC levels ------------------------------------------------
        emsec_list = []
        for i in range(1, 6):
            emsec_code = row.get(f'EMSEC{i}')
            if pd.notna(emsec_code):
                emsec_list.append({
                    'sector': row.get(f'EMSEC{i}_Sector', 'Unclassified') or 'Unclassified',
                    'industry': row.get(f'EMSEC{i}_Industry', 'Unclassified') or 'Unclassified',
                    'sub_industry': emsec_code
                })

        # ── 2) Collect EMTEC hierarchy ---------------------------------------------------
        lvl1 = parse_emtec_list(row.get('EMTEC_LEVEL1'))
        lvl2 = parse_emtec_list(row.get('EMTEC_LEVEL2'))
        lvl3 = parse_emtec_list(row.get('EMTEC_LEVEL3'))

        emtec_combos: List[dict] = []
        if lvl1:
            for l1 in lvl1:
                for l2 in (lvl2 or ['Unclassified']):
                    for l3 in (lvl3 or ['Unclassified']):
                        emtec_combos.append({'theme': l1, 'technology': l2, 'sub_technology': l3})
        else:
            emtec_combos.append({'theme': 'Unclassified', 'technology': 'Unclassified', 'sub_technology': 'Unclassified'})

        # ── 3) Produce cross‑product rows ----------------------------------------------
        for emsec in emsec_list:
            for emtec in emtec_combos:
                new_row = row.to_dict()
                new_row.update({
                    'Sector': emsec['sector'],
                    'Industry': emsec['industry'],
                    'Sub_industry': emsec['sub_industry'],
                    'Theme': emtec['theme'],
                    'Technology': emtec['technology'],
                    'Sub_Technology': emtec['sub_technology'],
                })
                expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)
