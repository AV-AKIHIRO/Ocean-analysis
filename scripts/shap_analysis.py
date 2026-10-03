#!/usr/bin/env python3
"""
scripts/shap_analysis.py
========================
Runs SHAP (SHapley Additive exPlanations) on Random Forest and XGBoost models
for both Salinity and CO2 prediction tasks. Produces:
  1. SHAP Summary (beeswarm) plots per target variable
  2. SHAP Bar plots (global feature importance)
  3. SHAP Dependence plots for top features
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
BASINS     = ['NA', 'SA', 'SO', 'PO', 'IO']


def run_shap_analysis(label):
    """Run SHAP analysis for a given target (Salinity or CO2)."""
    print(f"\n{'='*60}")
    print(f"  SHAP Analysis for {label} Prediction")
    print(f"{'='*60}")

    # --- Load Data ---
    X = pd.read_csv(os.path.join(OUTPUT_DIR, 'ml_features.csv'),
                    index_col=0, parse_dates=True)
    y = pd.read_csv(os.path.join(OUTPUT_DIR, f'ml_targets_{label.lower()}.csv'),
                    index_col=0, parse_dates=True)

    # Augment features with lagged targets (same as ml_feature_importance.py)
    y_arr = y.values
    y_lag = np.roll(y_arr, 1, axis=0)
    y_lag[0] = y_arr[0]
    X_aug = pd.DataFrame(
        np.hstack([X.values, y_lag]),
        columns=list(X.columns) + [f'{col}_lag' for col in y.columns]
    )

    # Target = increments (change per timestep)
    dy_arr = np.diff(y_arr, axis=0, prepend=y_arr[[0]])

    # --- Train Random Forest ---
    print(f"\n  Training Random Forest for {label}...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                               random_state=42, n_jobs=-1)
    rf.fit(X_aug, dy_arr)
    print(f"  RF Train R² = {rf.score(X_aug, dy_arr):.4f}")

    # --- Train XGBoost ---
    print(f"  Training XGBoost for {label}...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    )
    # XGBoost MultiOutput: train on mean of all basins for a single SHAP run
    dy_mean = dy_arr.mean(axis=1)
    xgb_model.fit(X_aug, dy_mean)
    print(f"  XGBoost Train R² = {xgb_model.score(X_aug, dy_mean):.4f}")

    # =====================================================================
    # SHAP for Random Forest (TreeExplainer)
    # =====================================================================
    print(f"\n  Computing SHAP values for Random Forest ({label})...")
    # Use multi-output: average SHAP across all basin outputs
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap_values = rf_explainer.shap_values(X_aug)

    # rf_shap_values: could be list of arrays or ndarray with shape variations
    if isinstance(rf_shap_values, list):
        # list of (n_samples, n_features), one per output
        rf_shap_avg = np.mean([np.abs(sv) for sv in rf_shap_values], axis=0)
        rf_shap_for_plot = rf_shap_values[0]  # First basin for beeswarm
    elif rf_shap_values.ndim == 3:
        # shape (n_samples, n_features, n_outputs) — average across outputs
        rf_shap_avg = np.abs(rf_shap_values).mean(axis=2)  # -> (n_samples, n_features)
        rf_shap_for_plot = rf_shap_values[:, :, 0]
    else:
        rf_shap_avg = np.abs(rf_shap_values)
        rf_shap_for_plot = rf_shap_values

    # --- RF SHAP Summary (Beeswarm) Plot ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(rf_shap_for_plot, X_aug, max_display=15, show=False)
    plt.title(f'SHAP Summary — Random Forest — {label} (First Basin)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_beeswarm = os.path.join(OUTPUT_DIR, f'shap_rf_summary_{label.lower()}.png')
    plt.savefig(out_beeswarm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_beeswarm}")

    # --- RF SHAP Bar Plot (Mean |SHAP|) ---
    plt.figure(figsize=(10, 6))
    mean_abs_shap = rf_shap_avg.mean(axis=0)  # -> (n_features,)
    feat_importance = pd.Series(mean_abs_shap, index=X_aug.columns).sort_values(ascending=False)
    feat_importance.head(15).plot(kind='barh', color='#1f77b4')
    plt.gca().invert_yaxis()
    plt.title(f'SHAP Global Feature Importance — Random Forest — {label}', fontsize=12, fontweight='bold')
    plt.xlabel('Mean |SHAP Value| (averaged across basins)')
    plt.tight_layout()
    out_bar = os.path.join(OUTPUT_DIR, f'shap_rf_bar_{label.lower()}.png')
    plt.savefig(out_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_bar}")

    # Print top features
    print(f"\n  RF SHAP Top 5 Features ({label}):")
    for feat, val in feat_importance.head(5).items():
        print(f"    {feat}: {val:.6f}")

    # =====================================================================
    # SHAP for XGBoost (TreeExplainer)
    # =====================================================================
    print(f"\n  Computing SHAP values for XGBoost ({label})...")
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(X_aug)

    # --- XGBoost SHAP Summary (Beeswarm) Plot ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(xgb_shap_values, X_aug, max_display=15, show=False)
    plt.title(f'SHAP Summary — XGBoost — {label} (Mean Basin)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_xgb_beeswarm = os.path.join(OUTPUT_DIR, f'shap_xgb_summary_{label.lower()}.png')
    plt.savefig(out_xgb_beeswarm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_xgb_beeswarm}")

    # --- XGBoost SHAP Bar Plot ---
    plt.figure(figsize=(10, 6))
    xgb_mean_abs = np.abs(xgb_shap_values).mean(axis=0)
    xgb_feat_importance = pd.Series(xgb_mean_abs, index=X_aug.columns).sort_values(ascending=False)
    xgb_feat_importance.head(15).plot(kind='barh', color='#ff7f0e')
    plt.gca().invert_yaxis()
    plt.title(f'SHAP Global Feature Importance — XGBoost — {label}', fontsize=12, fontweight='bold')
    plt.xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    out_xgb_bar = os.path.join(OUTPUT_DIR, f'shap_xgb_bar_{label.lower()}.png')
    plt.savefig(out_xgb_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_xgb_bar}")

    # --- XGBoost SHAP Dependence Plot for top 2 features ---
    top2 = xgb_feat_importance.head(2).index.tolist()
    for feat in top2:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, xgb_shap_values, X_aug, show=False)
        plt.title(f'SHAP Dependence — {feat} — XGBoost — {label}', fontsize=11, fontweight='bold')
        plt.tight_layout()
        safe_feat = feat.replace('/', '_')
        out_dep = os.path.join(OUTPUT_DIR, f'shap_xgb_dependence_{safe_feat}_{label.lower()}.png')
        plt.savefig(out_dep, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_dep}")

    print(f"\n  XGBoost SHAP Top 5 Features ({label}):")
    for feat, val in xgb_feat_importance.head(5).items():
        print(f"    {feat}: {val:.6f}")

    return feat_importance, xgb_feat_importance


def main():
    print("=" * 60)
    print("  SHAP (SHapley Additive exPlanations) Analysis")
    print("  Explaining Random Forest & XGBoost predictions")
    print("=" * 60)

    sal_rf, sal_xgb = run_shap_analysis('Salinity')
    co2_rf, co2_xgb = run_shap_analysis('CO2')

    print("\n" + "=" * 60)
    print("  ALL SHAP ANALYSES COMPLETE")
    print("=" * 60)
    print("\nGenerated plots in output/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('shap_'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
