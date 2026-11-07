"""P5 — Metrics and error-analysis helpers (used by src/train.py).

All metrics are on the Rupiah scale except R², which is on the log scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import config

METRICS = ("mae", "mape", "mdape", "r2_log")


def regression_metrics(y_true_idr: np.ndarray, y_pred_idr: np.ndarray) -> dict[str, float]:
    """MAE (Rp), MAPE (%), MdAPE (%), R² on log price."""
    y_true = np.asarray(y_true_idr, dtype=float)
    y_pred = np.asarray(y_pred_idr, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true * 100
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "mape": float(np.mean(ape)),
        "mdape": float(np.median(ape)),
        "r2_log": float(r2_score(np.log(y_true), np.log(y_pred))),
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each metric across folds."""
    df = pd.DataFrame(fold_metrics)
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_std"] = float(df[m].std(ddof=1))
    return out


def ape(y_true_idr: pd.Series, y_pred_idr: pd.Series) -> pd.Series:
    """Absolute percentage error per row (%)."""
    return (y_pred_idr - y_true_idr).abs() / y_true_idr * 100


def mape_by(oof: pd.DataFrame, column: str) -> pd.DataFrame:
    """MAPE and MdAPE by a grouping column of the out-of-fold table (needs 'ape')."""
    g = oof.groupby(column, observed=True)["ape"]
    return pd.DataFrame({"n": g.size(), "mape": g.mean(), "mdape": g.median()}).reset_index()


def price_band(price_idr: pd.Series) -> pd.Series:
    """Label each price with its config.PRICE_BANDS band."""
    return pd.cut(price_idr, bins=list(config.PRICE_BANDS), labels=list(config.PRICE_BAND_LABELS),
                  right=False)


def worst_predictions(oof: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n rows with the largest absolute percentage error."""
    return oof.sort_values("ape", ascending=False).head(n)


def sort_quantiles(pred_log: np.ndarray) -> np.ndarray:
    """Fix quantile crossing: sort each row's (q10, q50, q90) predictions."""
    return np.sort(np.atleast_2d(pred_log), axis=1)


def widen_log_interval(q_log: np.ndarray, factor: float | np.ndarray) -> np.ndarray:
    """Scale the distance of q10 and q90 from q50 by `factor` in log space (1.0 = unchanged).

    `factor` is one number or one per row (P7b U2 per-kabupaten scales).
    """
    q = np.array(np.atleast_2d(q_log), dtype=float)
    mid = q[:, 1]
    q[:, 0] = mid - factor * (mid - q[:, 0])
    q[:, 2] = mid + factor * (q[:, 2] - mid)
    return q
