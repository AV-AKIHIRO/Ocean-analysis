#!/usr/bin/env python3
"""
scripts/ml_feature_importance.py
=================================
Computes Gini and Permutation Feature Importance for Random Forest and XGBoost
models to analyze which satellite/physics features drive salinity and CO2 per basin.
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
BASINS     = ['NA', 'SA', 'SO', 'PO', 'IO']

def analyze_importance(label):
    print(f"\n--- Analyzing Feature Importance for {label} ---")
    X = pd.read_csv(os.path.join(OUTPUT_DIR, 'ml_features.csv'), index_col=0, parse_dates=True)
    y = pd.read_csv(os.path.join(OUTPUT_DIR, f'ml_targets_{label.lower()}.csv'), index_col=0, parse_dates=True)
    
    y_arr = y.values
    y_lag = np.roll(y_arr, 1, axis=0); y_lag[0] = y_arr[0]
    X_aug = pd.DataFrame(np.hstack([X.values, y_lag]), 
                         columns=list(X.columns) + [f'{col}_lag' for col in y.columns])
    dy_arr = np.diff(y_arr, axis=0, prepend=y_arr[[0]])

    # Fit RF and XGBoost
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_aug, dy_arr)
    
    rf_importances = pd.Series(rf.feature_importances_, index=X_aug.columns).sort_values(ascending=False)
    
    # Plot top 10 features
    plt.figure(figsize=(10, 5))
    rf_importances.head(10).plot(kind='barh', color='#2ca02c')
    plt.gca().invert_yaxis()
    plt.title(f'Random Forest Top Feature Importance — {label} Increments', fontsize=12, fontweight='bold')
    plt.xlabel('Relative Feature Importance (Gini)')
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f'feature_importance_{label.lower()}.png')
    plt.savefig(out_png, dpi=130)
    plt.close()
    print(f"Saved feature importance plot to {out_png}")
    print("Top 5 Features:")
    for feat, val in rf_importances.head(5).items():
        print(f"  - {feat}: {val:.4f}")

def main():
    analyze_importance('Salinity')
    analyze_importance('CO2')

if __name__ == '__main__':
    main()
