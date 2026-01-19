import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# =========================
# 0) Page config (new look)
# =========================
st.set_page_config(
    page_title="Likes Predictor (MLP)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# 1) Styling (different theme)
# =========================
st.markdown(
    """
    <style>
      /* ===== 强制全局深色背景（兼容新版本 Streamlit） ===== */
      html, body {
        background: #0B1220 !important;
      }
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(1200px 600px at 25% 10%, rgba(37,99,235,0.18), transparent 60%),
          radial-gradient(900px 500px at 80% 30%, rgba(147,51,234,0.14), transparent 55%),
          linear-gradient(180deg, #0B1220 0%, #0F172A 50%, #0B1220 100%) !important;
      }
      .stApp {
        background: transparent !important;  /* 避免 stApp 自己覆盖 */
      }

      /* 顶部 header/toolbar 透明，避免白条 */
      [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: transparent !important;
      }

      /* ===== 侧边栏深色 ===== */
      section[data-testid="stSidebar"] {
        background: #0F172A !important;
        border-right: 1px solid rgba(148,163,184,0.25);
      }

      /* ===== 字体颜色整体提亮（解决你说的灰字看不清） ===== */
      .stMarkdown, .stMarkdown p, .stText, .stCaption {
        color: rgba(234,240,255,0.90) !important;
      }
      label, .stTabs [data-baseweb="tab"] {
        color: rgba(234,240,255,0.78) !important;
      }
      .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
      }
      div[data-testid="stMetricLabel"] {
        color: rgba(234,240,255,0.78) !important;
      }
      div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
      }

      /* ===== 卡片/表格的底色，避免白底表格刺眼 ===== */
      .card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 16px 18px;
      }
      .muted {
        color: rgba(234,240,255,0.78) !important;
        font-size: 14px;
      }

      /* dataframe 容器也压暗（否则默认白底） */
      [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 12px !important;
      }

      /* st.info/st.warning 这类提示框的文字也提亮 */
      [data-testid="stAlert"] * {
        color: rgba(15, 23, 42, 0.95) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)



# =========================
# 2) Paths (edit here)
# =========================
BASE_DIR = Path(__file__).resolve().parent

# 改成你自己的目录：里面应包含 mlp_best.pt
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "mlp_best.pt"
META_PATH  = BASE_DIR / "models" / "mlp_meta.json"


# =========================
# 3) Model definition (must match training)
#    From your MLP_73.py: hidden_dims=(64,64), dropout=0.10 :contentReference[oaicite:3]{index=3}
# =========================
class MLPRegressor(nn.Module):
    def __init__(self, in_dim=3, hidden_dims=(64, 64), dropout=0.10):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _safe_expm1(x: float, clip_at: float = 50.0) -> float:
    """
    Prevent overflow in expm1 for extremely large values.
    clip_at=50 is already astronomically large in raw likes.
    """
    if x > clip_at:
        x = clip_at
    y = float(np.expm1(np.float64(x)))
    if not np.isfinite(y):
        return float("inf")
    return y


@st.cache_resource
def load_mlp_artifacts(model_path: Path, meta_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # meta is optional; if absent, use defaults consistent with MLP_73.py
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "feature_cols": ["Number of Fans", "Video Release Time", "Video Length"],
            "use_log_x": False,
            "use_log_y": True,            # your training uses y=log1p(likes) :contentReference[oaicite:4]{index=4}
            "hidden_dims": [64, 64],
            "dropout": 0.10
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(str(model_path), map_location=device)

    # ckpt keys per MLP_73.py: model_state_dict, scaler_mean, scaler_scale :contentReference[oaicite:5]{index=5}
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)

    hidden_dims = tuple(meta.get("hidden_dims", [64, 64]))
    dropout = float(meta.get("dropout", 0.10))

    model = MLPRegressor(in_dim=len(mean), hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, meta, mean, scale, device


def predict_likes_once(model, meta, mean, scale, device, fans, days, length):
    feature_cols = meta["feature_cols"]
    use_log_x = bool(meta.get("use_log_x", False))
    use_log_y = bool(meta.get("use_log_y", True))

    X_raw = np.array([[fans, days, length]], dtype=np.float32)

    # optional log1p on X (only if training did so)
    X_in = np.log1p(X_raw) if use_log_x else X_raw

    # StandardScaler: (x - mean) / scale
    X_s = (X_in - mean.reshape(1, -1)) / scale.reshape(1, -1)

    with torch.no_grad():
        xb = torch.from_numpy(X_s).to(device)
        pred = model(xb).detach().cpu().numpy().reshape(-1)[0]

    # pred is y_hat in log1p space if use_log_y=True
    if use_log_y:
        likes_hat = _safe_expm1(float(pred))
    else:
        likes_hat = float(pred)

    likes_hat = max(likes_hat, 0.0) if np.isfinite(likes_hat) else likes_hat

    return {
        "feature_cols": feature_cols,
        "use_log_x": use_log_x,
        "use_log_y": use_log_y,
        "pred_log1p": float(pred),
        "pred_raw": float(likes_hat),
    }


# =========================
# 4) Header (distributed text, not all in one place)
# =========================
left, right = st.columns([2.2, 1.0], gap="large")
with left:
    st.markdown(
        """
        <div class="card">
          <h3>Likes Predictor</h3>
          <div class="muted">
            MLP inference with the saved checkpoint (mlp_best.pt). Inputs are normalized using the scaler stats stored in the checkpoint.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="card">
          <div class="muted">Model</div>
          <div style="font-size:20px; font-weight:700;">MLP (PyTorch)</div>
          <div class="muted">Output: likes (raw)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# =========================
# 5) Sidebar inputs (new layout)
# =========================
st.sidebar.markdown("## Input Panel")
st.sidebar.caption("Inputs are fed into the MLP model after applying the same preprocessing used during training.")

with st.sidebar.form("input_form", clear_on_submit=False):
    fans = st.number_input("Number of Fans", min_value=0.0, value=10000.0, step=1000.0, format="%.0f")
    days = st.number_input("Video Release Time (days since upload)", min_value=0.0, value=30.0, step=1.0, format="%.0f")
    length = st.number_input("Video Length (seconds)", min_value=0.0, value=120.0, step=1.0, format="%.0f")

    #delta_pct = st.slider("Sensitivity probe (percent)", min_value=1, max_value=50, value=10, step=1)
    do_pred = st.form_submit_button("Run Prediction")


# =========================
# 6) Main area tabs (new layout)
# =========================
tab_pred, tab_info, tab_notes = st.tabs(["Prediction", "Model Details", "Notes"])

with tab_pred:
    st.markdown("### Prediction Output")
    st.caption("Results are shown in a summary card + auxiliary diagnostics. This layout differs from the old left/right columns UI.")

    if do_pred:
        try:
            model, meta, mean, scale, device = load_mlp_artifacts(MODEL_PATH, META_PATH)
            out = predict_likes_once(model, meta, mean, scale, device, fans, days, length)

            c1, c2 = st.columns(2, gap="large")
            c1.metric("Predicted Likes (raw)", f"{out['pred_raw']:,.0f}" if np.isfinite(out["pred_raw"]) else "inf")
            c2.metric("Device", device)

            st.markdown("")
            st.markdown(
                "<div class='card'><div class='muted'>Input summary</div></div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                {
                    "feature": ["Number of Fans", "Video Release Time", "Video Length"],
                    "value": [fans, days, length],
                },
                use_container_width=True,
                hide_index=True,
            )

            # Simple sensitivity probe: +/- delta% per feature (hold others fixed)


        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.markdown(
            """
            <div class="card">
            <div style="color: rgba(234,240,255,0.92); font-weight:600;">
                Set inputs in the left sidebar, then click <span style="color:#FFFFFF;">Run Prediction</span>.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_info:
    st.markdown("### Model Details")
    st.caption("This section is separate from the prediction view to avoid putting all text in one place.")

    st.write("Model file:", str(MODEL_PATH))
    st.write("Meta file (optional):", str(META_PATH))
    st.write("If meta is missing, defaults are used (3 features, y=log1p(likes)).")

    # Load and show meta if possible
    try:
        model, meta, mean, scale, device = load_mlp_artifacts(MODEL_PATH, META_PATH)
        st.json(meta)
        st.markdown("**Scaler mean/scale (from checkpoint):**")
        st.write("mean:", mean.tolist())
        st.write("scale:", scale.tolist())
    except Exception as e:
        st.warning(f"Could not load artifacts: {e}")

with tab_notes:
    st.markdown("### Notes")
    st.write(
        "- Ensure the MLP architecture here matches training (hidden dims, dropout). "
        "- Ensure preprocessing matches training: optional log1p(X) + StandardScaler + output expm1(y_hat)."
    )
    st.write(
        "If you change the training script hyperparameters (e.g., hidden layer sizes), "
        "you must update hidden_dims/dropout in meta or in this app."
    )
