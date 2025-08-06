import pandas as pd
import numpy as np

def compute_aggregate(sub_df, metric, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col):
    """Exact logic from standalone Heatmap (v3.6)."""
    if group_sel == "기업":
        if metric_main == "기업수":
            total = sub_df['Company'].nunique()
            if total == 0:
                return 0 if metric_mode != "결측 비율" else np.nan
            if metric_mode == "결측 포함":
                return total
            elif metric_mode == "결측 미포함":
                return sub_df.loc[~sub_df['has_missing_financials'], 'Company'].nunique()
            elif metric_mode == "결측 비율":
                missing = sub_df.loc[sub_df['has_missing_financials'], 'Company'].nunique()
                return missing / total if total else np.nan
        elif metric_main == "0이하비율":
            arr = pd.to_numeric(sub_df[base_col], errors="coerce")
            if metric_mode == "결측 제외":
                arr = arr.dropna()
            return (arr <= 0).sum() / len(arr) if len(arr) else np.nan

    else:  # 멀티플·재무비율
        if agg_func == "AGG":
            mc_col = 'Market Cap (2024-12-31)_USD'
            if mc_col not in sub_df.columns or sub_df[mc_col].isna().all():
                return np.nan
            mc = sub_df[mc_col]
            q1, q3 = mc.quantile(0.25), mc.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 2*iqr, q3 + 2*iqr
            filt = sub_df[(mc >= lower) & (mc <= upper)]
            if filt.empty:
                return np.nan
            if metric in ['PER', 'PBR', 'EV_EBITDA']:
                if metric == 'PER':
                    num, den = filt[mc_col].sum(), filt['Net_Income'].sum()
                elif metric == 'PBR':
                    num, den = filt[mc_col].sum(), filt['Book'].sum()
                else:  # EV_EBITDA
                    num, den = filt['Enterprise Value (FQ0)_USD'].sum(), filt['EBITDA'].sum()
                return num / den if den else np.nan
            arr = filt[metric].dropna()
            return arr.sum() if len(arr) else np.nan
        else:  # AVG / MED / HRM
            arr = sub_df[metric].dropna()
            if not len(arr):
                return np.nan
            if agg_func == 'AVG':
                return arr.mean()
            elif agg_func == 'MED':
                return arr.median()
            else:  # HRM
                arr = arr[arr > 0]
                return len(arr) / (1/arr).sum() if len(arr) else np.nan

# ---------------------------------------------------------------------------
