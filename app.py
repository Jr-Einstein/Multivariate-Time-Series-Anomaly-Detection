# app.py
# Streamlit demo: LSTM-AE anomaly detection for NASA CMAPSS FD001
# Uses artifacts saved by 02_anomaly_if_lstm.ipynb

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ARTIFACTS = Path("artifacts_fd001")

# ---------- Load artifacts ----------
@st.cache_data(show_spinner=False)
def load_artifacts():
    req = [
        "X.npy",
        "meta.npy",
        "healthy_win.npy",
        "score_smooth_lstm.npy",
        "alerts_lstm.npy",
        "feature_names.npy",
        "threshold_lstm.json",
    ]
    for f in req:
        if not (ARTIFACTS / f).exists():
            raise FileNotFoundError(
                f"Missing {f} in {ARTIFACTS}. "
                "Re-run the notebook to save artifacts."
            )

    X = np.load(ARTIFACTS / "X.npy")                # (N, W, F)
    meta = np.load(ARTIFACTS / "meta.npy")          # (N, 2) = [engine_id, cycle_end]
    healthy_win = np.load(ARTIFACTS / "healthy_win.npy").astype(bool)
    score = np.load(ARTIFACTS / "score_smooth_lstm.npy").astype(float).ravel()
    alerts = np.load(ARTIFACTS / "alerts_lstm.npy").astype(bool).ravel()
    feat_names = np.load(ARTIFACTS / "feature_names.npy", allow_pickle=True).tolist()
    with open(ARTIFACTS / "threshold_lstm.json", "r") as f:
        thr = float(json.load(f)["thr_lstm"])

    # baseline mu/sd for contribution analysis (last timestep of healthy windows)
    last_healthy = X[healthy_win, -1, :]  # (n_healthy, F)
    mu = last_healthy.mean(axis=0)
    sd = last_healthy.std(axis=0) + 1e-8

    return {
        "X": X,
        "meta": meta,
        "healthy_win": healthy_win,
        "score": score,
        "alerts": alerts,
        "thr": thr,
        "feat_names": feat_names,
        "mu": mu,
        "sd": sd,
    }

A = load_artifacts()
X = A["X"]; meta = A["meta"]; healthy_win = A["healthy_win"]
score = A["score"]; alerts = A["alerts"]; thr = A["thr"]
feat_names = A["feat_names"]; mu = A["mu"]; sd = A["sd"]

W = X.shape[1]
F = X.shape[2]
engine_ids = np.unique(meta[:, 0]).astype(int)

# ---------- Small helpers ----------
def consecutive_alerts(alerts_vec: np.ndarray, meta_arr: np.ndarray, k: int = 3) -> np.ndarray:
    """Return alerts that require >=k consecutive True within each engine."""
    out = alerts_vec.copy()
    for uid in np.unique(meta_arr[:, 0]):
        m = meta_arr[:, 0] == uid
        seq = out[m].astype(int)
        if len(seq) >= k:
            # 'same' keeps alignment
            streak = np.convolve(seq, np.ones(k, dtype=int), mode="same") >= k
            out[m] = streak
    return out

def first_alert_idx_for_engine(uid: int, alerts_vec: np.ndarray, meta_arr: np.ndarray):
    m = (meta_arr[:, 0] == uid) & alerts_vec
    idxs = np.where(m)[0]
    return int(idxs[0]) if len(idxs) else None

def topk_contrib_for_idx(idx: int, k: int = 5):
    """Top-k sensors by |z-score| at the last timestep of window idx."""
    last = X[idx, -1, :]                  # (F,)
    z = np.abs((last - mu) / sd)          # (F,)
    order = np.argsort(z)[::-1][:k]
    names = [feat_names[i] if i < len(feat_names) else f"feat_{i}" for i in order]
    return names, z[order]

# ---------- UI ----------
st.set_page_config(page_title="Turbofan Anomaly Demo", layout="wide")
st.title("✈️ Turbofan Engine Anomaly Detection (LSTM Autoencoder)")

with st.sidebar:
    st.header("Controls")
    uid = st.selectbox("Engine", options=engine_ids, index=0)
    k_cons = st.slider("Alert smoothing (k consecutive breaches)", min_value=1, max_value=5, value=3, step=1)
    show_contrib = st.checkbox("Show top contributing sensors at first alert", value=True)
    ctx = st.slider("Context window (± cycles) for sensor traces", 5, 40, 15, 1)

# choose alerts version
alerts_k = consecutive_alerts(alerts, meta, k=k_cons) if k_cons > 1 else alerts

# ---------- Plot anomaly score ----------
m_uid = meta[:, 0] == uid
cyc_w = meta[m_uid, 1]
s = score[m_uid]
a = alerts_k[m_uid]

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(cyc_w, s, label="Anomaly score")
ax.axhline(thr, ls="--", color="tab:blue", alpha=0.7, label="Threshold")
if a.any():
    ax.scatter(cyc_w[a], s[a], s=18, color="tab:red", label="Alerts", zorder=3)
ax.set_xlabel("cycle (window end)")
ax.set_ylabel("score")
ax.set_title(f"Engine {uid} — LSTM-AE anomaly score")
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

# ---------- Contributors at first alert ----------
if show_contrib:
    idx = first_alert_idx_for_engine(uid, alerts_k, meta)
    if idx is None:
        st.info("No alert for this engine with the current smoothing. Try lowering k, or choose another engine.")
    else:
        names, zs = topk_contrib_for_idx(idx, k=5)
        st.subheader(f"Top contributing sensors at first alert (engine {uid})")

        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(names, zs)
        ax2.set_ylabel("Deviation (z-score)")
        ax2.set_title("Top-5 sensors (higher = more abnormal)")
        st.pyplot(fig2, clear_figure=True)

        # Context traces (last step of each window; normalized z if scales differ)
        st.caption("Context around first alert (normalized z-scores at window end)")
        # find local indices within engine
        uid_idx_all = np.where(m_uid)[0]
        local_pos = np.where(uid_idx_all == np.where((meta[:, 0] == uid) & alerts_k)[0][0])[0][0]
        start = max(0, local_pos - ctx)
        end = min(len(uid_idx_all) - 1, local_pos + ctx)
        cyc_ctx = cyc_w[start : end + 1]

        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        for nm in names:
            f_idx = feat_names.index(nm) if nm in feat_names else int(nm.split("_")[-1])
            series = [ (X[i, -1, f_idx] - mu[f_idx]) / sd[f_idx] for i in uid_idx_all[start : end + 1] ]
            ax3.plot(cyc_ctx, series, label=nm)
        ax3.axvline(cyc_w[local_pos], ls="--", color="k", alpha=0.6, label="first alert")
        ax3.set_xlabel("cycle"); ax3.set_ylabel("z-score")
        ax3.set_title("Top sensors around alert")
        ax3.legend(ncols=3, fontsize=9)
        st.pyplot(fig3, clear_figure=True)

# ---------- Footer ----------
st.divider()
st.caption(
    f"Windows: {X.shape[0]}  |  W={W}  |  F={F}  ·  "
    f"Healthy windows: {healthy_win.sum()} ({healthy_win.mean():.1%})  ·  "
    f"Threshold (healthy p95): {thr:.4g}"
)
