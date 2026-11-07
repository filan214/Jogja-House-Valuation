"""P6b — Explanations for one prediction, plus the offline SHAP summary figure.

`explain(features_row)` uses LightGBM's built-in SHAP values
(`booster.predict(X, pred_contrib=True)`, log-price units, last column = bias),
sums them per human label in config.FEATURE_LABELS, and converts each group
sum to an approximate % effect: (exp(c) − 1) × 100. App-safe: imports only
numpy, pandas and lightgbm; shap and matplotlib are imported by the figure code.

Offline output: reports/figures/09_shap_summary.png (beeswarm on the point model)

Run: python -m src.explain
"""

from __future__ import annotations

from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_point_booster() -> lgb.Booster:
    """The saved LightGBM point model (loaded once)."""
    return lgb.Booster(model_file=str(config.LGBM_POINT_MODEL))


def contributions(features: pd.DataFrame, booster: lgb.Booster) -> pd.DataFrame:
    """Per-feature SHAP contributions in log-price units; column 'bias' = expected value."""
    contrib = booster.predict(features[booster.feature_name()], pred_contrib=True)
    return pd.DataFrame(contrib, columns=[*booster.feature_name(), "bias"], index=features.index)


def grouped_contributions(contrib_row: pd.Series) -> pd.Series:
    """Sum one row's feature contributions per config.FEATURE_LABELS label (log units)."""
    feats = contrib_row.drop("bias")
    missing = set(feats.index) - set(config.FEATURE_LABELS)
    if missing:
        raise KeyError(f"no FEATURE_LABELS entry for {sorted(missing)}")
    return feats.groupby(feats.index.map(config.FEATURE_LABELS)).sum()


def explain(features_row: pd.DataFrame, booster: lgb.Booster | None = None,
            top: int = config.N_DRIVERS) -> list[dict]:
    """Top `top` drivers of one prediction: [{"label": str, "pct_effect": float}, ...].

    pct_effect is the approximate % change in the estimate from this group of
    features relative to an average listing, e.g. +25.0 = about 25% higher.
    """
    if len(features_row) != 1:
        raise ValueError(f"explain() expects one row, got {len(features_row)}")
    booster = booster or load_point_booster()
    groups = grouped_contributions(contributions(features_row, booster).iloc[0])
    order = groups.abs().sort_values(ascending=False).index[:top]
    return [{"label": str(lbl), "pct_effect": float((np.exp(groups[lbl]) - 1) * 100)} for lbl in order]


def display_name(feature: str) -> str:
    """Readable name for figure 09, e.g. 'Land size (log_land)'."""
    label = config.FEATURE_LABELS.get(feature, feature)
    return label if feature in config.ACTIVE_FLAGS else f"{label} ({feature})"


def fig_shap_summary(X: pd.DataFrame, booster: lgb.Booster) -> str:
    """09 — SHAP beeswarm of the point model on all training rows."""
    import matplotlib.pyplot as plt
    import shap
    from matplotlib.colors import LinearSegmentedColormap

    from src.eda import C, _caption, _save, _style

    _style()
    values = contributions(X, booster)
    expl = shap.Explanation(values=values.drop(columns="bias").to_numpy(),
                            base_values=values["bias"].to_numpy(), data=X.to_numpy(),
                            feature_names=[display_name(f) for f in X.columns])
    cmap = LinearSegmentedColormap.from_list("feature_value", [C["series_1"], C["context"], C["series_2"]])
    shap.plots.beeswarm(expl, max_display=config.SHAP_MAX_DISPLAY, color=cmap, show=False, plot_size=(9, 7))
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title("What moves the estimate — SHAP values of the point model", loc="left")
    ax.set_xlabel("Effect on log price (0.1 ≈ +10.5% price)")
    _caption(fig, "Each dot is one listing. Right of 0 = pushes the estimate up. Color = the feature's value "
                  "(orange high, blue low, dark gray missing). Flags are 0/1.")
    return _save(fig, "09_shap_summary.png")


def main() -> None:
    """Draw figure 09 and print a sanity check plus an example explanation."""
    table = pd.read_csv(config.MODEL_TABLE, dtype={config.CV_GROUP_COLUMN: str})
    booster = load_point_booster()
    X = table[booster.feature_name()]
    contrib = contributions(X, booster)
    raw = booster.predict(X, raw_score=True)
    max_gap = float(np.max(np.abs(contrib.sum(axis=1).to_numpy() - raw)))
    print(f"{len(X):,} rows; contributions sum to the raw prediction (max gap {max_gap:.2e})")
    mean_abs = contrib.drop(columns="bias").abs().mean().groupby(
        lambda f: config.FEATURE_LABELS[f]).sum().sort_values(ascending=False)
    print("Mean |contribution| per group (log units):")
    for lbl, v in mean_abs.items():
        print(f"  {lbl:<34} {v:.3f}")
    i = int((table["price_idr"] - table["price_idr"].median()).abs().idxmin())
    print(f"Example (median-priced listing {table.loc[i, 'listing_id']}, {table.loc[i, 'kecamatan']}):")
    for d in explain(X.loc[[i]], booster):
        print(f"  {d['label']:<34} {d['pct_effect']:+6.1f}%")
    print(f"Figure: {fig_shap_summary(X, booster)}")


if __name__ == "__main__":
    main()
