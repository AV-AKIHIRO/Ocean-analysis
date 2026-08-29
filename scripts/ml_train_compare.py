#!/usr/bin/env python3
"""
ml_train_compare.py  (v2 — autoregressive differencing)
=========================================================
Trains 6 ML models to predict per-basin INCREMENTAL CHANGES in
salinity ΔS_i(t) and CO₂ ΔC_i(t), then cumulates to get trajectories.

Key fix from v1: models predict Δy(t) = y(t) − y(t-1), using y(t-1)
as an additional feature. This mirrors the ODE's "integrate forward"
mechanism and allows genuine out-of-sample trend extrapolation.

Models:
  1. Linear Regression  (baseline)
  2. Ridge Regression
  3. Random Forest
  4. Gradient Boosting (XGBoost)
  5. LSTM  (PyTorch — 12-month window, predicts next Δy)
  6. Transformer Encoder (PyTorch)
"""

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(WORKSPACE, 'output')
ARTIFACT_DIR = '/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63'
BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']
LOOK_BACK = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")

MODEL_COLORS = {
    'ODE': '#000000', 'Linear': '#e41a1c', 'Ridge': '#377eb8',
    'RandomForest': '#4daf4a', 'XGBoost': '#984ea3',
    'LSTM': '#ff7f00', 'Transformer': '#a65628',
}
MODEL_STYLES = {
    'ODE': '-', 'Linear': '--', 'Ridge': ':', 'RandomForest': '-.',
    'XGBoost': '--', 'LSTM': '-', 'Transformer': ':'
}
COLORS = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd']

# ════════════════════════════════════════════════════════════════
# PYTORCH MODELS
# ════════════════════════════════════════════════════════════════
class LSTMModel(nn.Module):
    def __init__(self, n_feat, n_out, hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_out))
    def forward(self, x):
        out, _ = self.lstm(x); return self.head(out[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, n_feat, n_out, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, n_out))
    def forward(self, x):
        x = self.input_proj(x); x = self.encoder(x); return self.head(x[:, -1, :])

def make_sequences(X_arr, dy_arr, look_back):
    Xs, ys = [], []
    for i in range(look_back, len(X_arr)):
        Xs.append(X_arr[i-look_back:i])
        ys.append(dy_arr[i])
    return np.array(Xs), np.array(ys)

def train_pytorch(model, X_tr, y_tr, X_va, y_va, epochs=500, lr=1e-3, patience=50):
    XT = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yT = torch.tensor(y_tr, dtype=torch.float32).to(device)
    dl = DataLoader(TensorDataset(XT, yT), batch_size=32, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    loss_fn = nn.MSELoss()
    best_loss = np.inf; best_state = None; no_imp = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(torch.tensor(X_va, dtype=torch.float32).to(device)),
                torch.tensor(y_va, dtype=torch.float32).to(device)).item()
        sched.step(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss; no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= patience: break
    model.load_state_dict(best_state); return model

def pt_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

# ════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred, cols):
    rows = []
    for i, c in enumerate(cols):
        yt = y_true[:, i]; yp = y_pred[:, i]
        rows.append({'basin': c, 'R2': r2_score(yt, yp),
                     'RMSE': np.sqrt(mean_squared_error(yt, yp)),
                     'MAE':  mean_absolute_error(yt, yp)})
    return pd.DataFrame(rows)

# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE — AUTOREGRESSIVE DIFFERENCING
# ════════════════════════════════════════════════════════════════
def run_pipeline(X, y, meta, label, unit):
    """
    Autoregressive differencing:
      - Augment X with y(t-1) as additional features
      - Target = Δy(t) = y(t) - y(t-1)
      - Reconstruct trajectory by cumsum from first known value
    """
    n_out     = y.shape[1]
    col_names = list(y.columns)
    dates     = X.index
    scaler_X  = meta['scaler_X']
    tr = meta['train_mask']; te = meta['test_mask']

    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)

    # ── Build augmented feature matrix: X + y_lag ─────────────────
    y_lag = np.roll(y_arr, 1, axis=0); y_lag[0] = y_arr[0]   # lag-1 target
    X_aug = np.hstack([X_arr, y_lag])   # (N, 14+n_out)

    # ── Differenced target ────────────────────────────────────────
    dy_arr = np.diff(y_arr, axis=0, prepend=y_arr[[0]])   # Δy(t)

    # ── Scale ─────────────────────────────────────────────────────
    scaler_Xa  = StandardScaler()
    # Per-column scaler for dy: critical for CO₂ where SO (Δ≈0.004) >> PO (Δ≈0.00003)
    scalers_dy = [StandardScaler() for _ in range(n_out)]

    X_tr_sc  = scaler_Xa.fit_transform(X_aug[tr])
    X_te_sc  = scaler_Xa.transform(X_aug[te])

    # Fit per-column scalers on training dy
    dy_tr_sc = np.column_stack([
        scalers_dy[i].fit_transform(dy_arr[tr, i:i+1]).ravel()
        for i in range(n_out)])

    def scale_dy(arr):
        return np.column_stack([scalers_dy[i].transform(arr[:, i:i+1]).ravel()
                                 for i in range(n_out)])
    def unscale_dy(arr):
        return np.column_stack([scalers_dy[i].inverse_transform(arr[:, i:i+1]).ravel()
                                 for i in range(n_out)])

    # Sequence data (scaled over full array)
    X_aug_all_sc = scaler_Xa.transform(X_aug)
    dy_all_sc    = scale_dy(dy_arr)
    Xs_all, dys_all = make_sequences(X_aug_all_sc, dy_all_sc, LOOK_BACK)


    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    seq_tr = np.array([i - LOOK_BACK for i in tr_idx if i >= LOOK_BACK])
    seq_te = np.array([i - LOOK_BACK for i in te_idx if i >= LOOK_BACK])
    Xs_tr = Xs_all[seq_tr]; dys_tr = dys_all[seq_tr]
    Xs_te = Xs_all[seq_te]
    dy_te_raw = dy_arr[te_idx[te_idx >= LOOK_BACK]]
    y_te_raw  = y_arr[te]
    dates_te_seq = dates[te_idx[te_idx >= LOOK_BACK]]
    n_feat = X_aug.shape[1]

    # ── Helper: reconstruct trajectory from Δy predictions ────────
    def reconstruct(dy_pred_raw, start_val, full=False):
        """
        Integrate Δy predictions forward from start_val.
        dy_pred_raw: (T, n_out) array of predicted increments
        start_val:   (n_out,) last known y value before forecast starts
        """
        traj = np.zeros_like(dy_pred_raw)
        traj[0] = start_val + dy_pred_raw[0]
        for t in range(1, len(dy_pred_raw)):
            traj[t] = traj[t-1] + dy_pred_raw[t]
        return traj

    results = {}
    start_val_te = y_arr[tr][-1]   # last training value → seed for test reconstruction

    # ── 1. Linear Regression ──────────────────────────────────────
    print(f"  [1/6] Linear Regression...", end=' ', flush=True)
    m = LinearRegression()
    m.fit(X_tr_sc, dy_tr_sc)
    dy_pred_tr = unscale_dy(m.predict(X_tr_sc))
    dy_pred_te = unscale_dy(m.predict(X_te_sc))
    y_pred_te  = reconstruct(dy_pred_te, start_val_te)
    r2 = metrics(y_te_raw, y_pred_te, col_names)
    results['Linear'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te, m_te=r2, sklearn_model=m)
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── 2. Ridge Regression ───────────────────────────────────────
    print(f"  [2/6] Ridge Regression...", end=' ', flush=True)
    m = Ridge(alpha=1.0)
    m.fit(X_tr_sc, dy_tr_sc)
    dy_pred_te = unscale_dy(m.predict(X_te_sc))
    y_pred_te  = reconstruct(dy_pred_te, start_val_te)
    r2 = metrics(y_te_raw, y_pred_te, col_names)
    results['Ridge'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te, m_te=r2, sklearn_model=m)
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── 3. Random Forest ──────────────────────────────────────────
    print(f"  [3/6] Random Forest...", end=' ', flush=True)
    m = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=8,
                                                    random_state=42, n_jobs=-1))
    m.fit(X_tr_sc, dy_arr[tr])
    dy_pred_te = m.predict(X_te_sc)
    y_pred_te  = reconstruct(dy_pred_te, start_val_te)
    r2 = metrics(y_te_raw, y_pred_te, col_names)
    results['RandomForest'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te, m_te=r2, sklearn_model=m)
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── 4. XGBoost ────────────────────────────────────────────────
    print(f"  [4/6] XGBoost...", end=' ', flush=True)
    m = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42, verbosity=0))
    m.fit(X_tr_sc, dy_arr[tr])
    dy_pred_te = m.predict(X_te_sc)
    y_pred_te  = reconstruct(dy_pred_te, start_val_te)
    r2 = metrics(y_te_raw, y_pred_te, col_names)
    results['XGBoost'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te, m_te=r2, sklearn_model=m)
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── 5. LSTM ───────────────────────────────────────────────────
    print(f"  [5/6] LSTM...", end=' ', flush=True)
    va_split  = int(0.8 * len(Xs_tr))
    lstm = LSTMModel(n_feat, n_out).to(device)
    lstm = train_pytorch(lstm, Xs_tr[:va_split], dys_tr[:va_split],
                          Xs_tr[va_split:], dys_tr[va_split:], epochs=600, patience=60)
    dy_pred_sc = pt_predict(lstm, Xs_te)
    dy_pred_te = unscale_dy(dy_pred_sc)
    # Reconstruct from last training val (LOOK_BACK offset)
    start_seq  = y_arr[te_idx[LOOK_BACK-1]] if LOOK_BACK <= len(te_idx) else start_val_te
    y_pred_te_seq = reconstruct(dy_pred_te, start_seq)
    r2 = metrics(y_arr[te_idx[te_idx >= LOOK_BACK]], y_pred_te_seq, col_names)
    results['LSTM'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te_seq,
                            m_te=r2, model=lstm, dates_te=dates_te_seq)
    torch.save(lstm.state_dict(), os.path.join(OUTPUT_DIR, f'lstm_{label}.pt'))
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── 6. Transformer ────────────────────────────────────────────
    print(f"  [6/6] Transformer...", end=' ', flush=True)
    trf = TransformerModel(n_feat, n_out).to(device)
    trf = train_pytorch(trf, Xs_tr[:va_split], dys_tr[:va_split],
                         Xs_tr[va_split:], dys_tr[va_split:], epochs=600, patience=60)
    dy_pred_sc = pt_predict(trf, Xs_te)
    dy_pred_te = unscale_dy(dy_pred_sc)
    y_pred_te_seq = reconstruct(dy_pred_te, start_seq)
    r2 = metrics(y_arr[te_idx[te_idx >= LOOK_BACK]], y_pred_te_seq, col_names)
    results['Transformer'] = dict(dy_pred_te=dy_pred_te, y_pred_te=y_pred_te_seq,
                                   m_te=r2, model=trf, dates_te=dates_te_seq)
    torch.save(trf.state_dict(), os.path.join(OUTPUT_DIR, f'transformer_{label}.pt'))
    print(f"R2_test={r2['R2'].mean():.3f}")

    # ── Save ──────────────────────────────────────────────────────
    save_obj = {k: {kk: vv for kk, vv in v.items() if kk not in ('model', 'sklearn_model')}
                for k, v in results.items()}
    with open(os.path.join(OUTPUT_DIR, f'ml_results_{label}.pkl'), 'wb') as f:
        pickle.dump({'results': save_obj,
                     'dates_tr': dates[tr], 'dates_te': dates[te],
                     'dates_te_seq': dates_te_seq,
                     'y_tr': y_arr[tr], 'y_te': y_te_raw,
                     'col_names': col_names, 'unit': unit,
                     'scaler_Xa': scaler_Xa, 'scaler_dy': scalers_dy,
                     'tr_idx': tr_idx, 'te_idx': te_idx,
                     'Xs_all': Xs_all, 'X_aug_all_sc': X_aug_all_sc,
                     'dy_arr': dy_arr, 'y_arr': y_arr,
                     'n_feat': n_feat, 'n_out': n_out,
                     'start_val_te': start_val_te}, f)

    return results, dates, y_arr, tr, te, dates_te_seq, dy_arr

# ════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════
def plot_timeseries(results, dates, y_all, tr, te, label, unit, col_names):
    dt_all = pd.DatetimeIndex(dates)
    dt_tr  = dt_all[tr]; dt_te = dt_all[te]
    n_b = len(col_names)
    fig, axes = plt.subplots(n_b, 1, figsize=(16, 4*n_b), sharex=True)
    fig.suptitle(f'ML vs ODE: {label} — Autoregressive Models (1987–2014)\n'
                 f'All models predict Δ{label}(t), reconstructed via cumulative sum',
                 fontsize=12, fontweight='bold', y=1.001)
    for bi, (ax, col) in enumerate(zip(axes, col_names)):
        basin = col.split('_')[1]
        ax.plot(dt_all, y_all[:, bi], color='black', lw=2.0, label='ODE (physics)', zorder=10)
        ax.axvline(dt_te[0], color='gray', ls='--', lw=1.0, alpha=0.7)
        for mname, res in results.items():
            if mname in ('LSTM','Transformer'):
                dt_use = pd.DatetimeIndex(res.get('dates_te', dt_te))
            else:
                dt_use = dt_te
            pred = res['y_pred_te']
            ax.plot(dt_use[:len(pred)], pred[:, bi],
                    color=MODEL_COLORS[mname], ls=MODEL_STYLES[mname],
                    lw=1.6, label=f'{mname}', alpha=0.85)
        ax.set_ylabel(f'{label} [{unit}]', fontsize=9)
        ax.set_title(basin, fontsize=9, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        if bi == 0:
            ax.legend(fontsize=8, ncol=4, loc='upper left')
    axes[-1].set_xlabel('Year', fontsize=11)
    plt.tight_layout()
    for path in [os.path.join(OUTPUT_DIR, f'ml_timeseries_{label.lower()}.png'),
                 os.path.join(ARTIFACT_DIR, f'ml_timeseries_{label.lower()}.png')]:
        plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved ml_timeseries_{label.lower()}.png")

def plot_r2_heatmap(results, col_names, label):
    model_names = list(results.keys())
    basins = [c.split('_')[1] for c in col_names]
    R2 = np.zeros((len(model_names), len(basins)))
    for mi, mn in enumerate(model_names):
        r2s = results[mn]['m_te'].set_index('basin')['R2']
        for bi, col in enumerate(col_names):
            R2[mi, bi] = r2s.get(col, np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(R2, vmin=-0.5, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(basins))); ax.set_xticklabels(basins, fontsize=11)
    ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names, fontsize=11)
    plt.colorbar(im, ax=ax, label='Test R²')
    for mi in range(len(model_names)):
        for bi in range(len(basins)):
            v = R2[mi, bi]
            ax.text(bi, mi, f'{v:.2f}', ha='center', va='center', fontsize=9,
                    color='black' if -0.2 < v < 0.8 else 'white')
    ax.set_title(f'Test R² Heatmap — {label} (Reconstructed Trajectory)\nAll Models × All Basins',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    for path in [os.path.join(OUTPUT_DIR, f'ml_r2_heatmap_{label.lower()}.png'),
                 os.path.join(ARTIFACT_DIR, f'ml_r2_heatmap_{label.lower()}.png')]:
        plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved ml_r2_heatmap_{label.lower()}.png")

def plot_residuals(results, dates, y_all, te, label, unit, col_names):
    dt_te = pd.DatetimeIndex(dates[te])
    y_te  = y_all[te]
    n_b   = len(col_names)
    fig, axes = plt.subplots(n_b, 1, figsize=(16, 3*n_b), sharex=True)
    fig.suptitle(f'Residuals (ML − ODE) on Test Set (2010–2014): {label}',
                 fontsize=12, fontweight='bold', y=1.001)
    for bi, (ax, col) in enumerate(zip(axes, col_names)):
        basin = col.split('_')[1]
        ax.axhline(0, color='black', lw=0.9, ls='--')
        for mname, res in results.items():
            pred = res['y_pred_te']
            if mname in ('LSTM','Transformer'):
                dt_use = pd.DatetimeIndex(res.get('dates_te', dt_te))
                y_ref  = y_all[np.isin(dates, dt_use)]
            else:
                dt_use = dt_te; y_ref = y_te
            resid = pred[:len(y_ref), bi] - y_ref[:len(pred), bi]
            ax.plot(dt_use[:len(resid)], resid,
                    color=MODEL_COLORS[mname], ls=MODEL_STYLES[mname],
                    lw=1.3, label=mname, alpha=0.85)
        ax.set_ylabel(f'Δ{label} [{unit}]', fontsize=9)
        ax.set_title(basin, fontsize=9, fontweight='bold')
        ax.grid(True, ls='--', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if bi == 0:
            ax.legend(fontsize=8, ncol=6, loc='upper right')
    axes[-1].set_xlabel('Year', fontsize=11)
    plt.tight_layout()
    for path in [os.path.join(OUTPUT_DIR, f'ml_residuals_{label.lower()}.png'),
                 os.path.join(ARTIFACT_DIR, f'ml_residuals_{label.lower()}.png')]:
        plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved ml_residuals_{label.lower()}.png")

def print_table(results, label):
    print(f"\n{'='*70}\n  TEST METRICS — {label}\n{'='*70}")
    print(f"  {'Model':<16} {'Mean R²':>9} {'Mean RMSE':>11} {'Mean MAE':>10}")
    print(f"  {'-'*50}")
    for mn, res in results.items():
        m = res['m_te']
        print(f"  {mn:<16} {m['R2'].mean():>9.4f} {m['RMSE'].mean():>11.6f} {m['MAE'].mean():>10.6f}")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print('='*65)
    print('ML Pipeline v2 — Autoregressive Differencing')
    print('6 Models × 2 Targets × 5 Basins')
    print('='*65)
    X     = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_features.csv'), index_col=0, parse_dates=True)
    y_sal = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_targets_salinity.csv'), index_col=0, parse_dates=True)
    y_co2 = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_targets_co2.csv'), index_col=0, parse_dates=True)
    with open(os.path.join(OUTPUT_DIR,'ml_meta.pkl'),'rb') as f:
        meta = pickle.load(f)

    print('\n[1/2] SALINITY')
    res_sal, dates, y_sal_arr, tr, te, dt_te_seq, dy_sal = \
        run_pipeline(X, y_sal, meta, 'Salinity', 'psu')
    print_table(res_sal, 'Salinity')
    plot_timeseries(res_sal, dates, y_sal_arr, tr, te, 'Salinity', 'psu', list(y_sal.columns))
    plot_r2_heatmap(res_sal, list(y_sal.columns), 'Salinity')
    plot_residuals(res_sal, dates, y_sal_arr, te, 'Salinity', 'psu', list(y_sal.columns))

    print('\n[2/2] CO2')
    res_co2, dates, y_co2_arr, tr, te, dt_te_seq2, dy_co2 = \
        run_pipeline(X, y_co2, meta, 'CO2', 'mmol/m3')
    print_table(res_co2, 'CO2')
    plot_timeseries(res_co2, dates, y_co2_arr, tr, te, 'CO2', 'mmol/m3', list(y_co2.columns))
    plot_r2_heatmap(res_co2, list(y_co2.columns), 'CO2')
    plot_residuals(res_co2, dates, y_co2_arr, te, 'CO2', 'mmol/m3', list(y_co2.columns))

    print('\nAll done.')

if __name__ == '__main__':
    main()
