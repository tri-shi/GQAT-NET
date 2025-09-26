# -*- coding: utf-8 -*-
"""
End-to-end GQAT-Net on dfas.csv (Alibaba ACT)
- Preprocessing per paper + 3 time-series features from 'start_date'
  -> ts_hour, ts_dayofweek, ts_month
- Feature selection: Boruta + MI + DFS (use the UNION of all selected features; NO 24-feature cap)
- 5-fold CV with SMOTE on train
- Isotonic calibration + recalibration analysis (NO calibration plots saved)
- MC-Dropout uncertainty
- Saves metrics/plots/models to ./results and ./plots
- Reports model size/params and computational footprint during train & inference:
  * CPU% (avg/max), peak RAM (MB)
  * GPU util% / peak GPU memory (MB) if nvidia-smi available (fallback to TF memory info)

Fixed hyperparameters (per paper):
  Batch Size = 128
  Learning Rate = 1e-4
  Epochs = 80
  Dropout = 0.3
  Attention Heads = 4
  Capsule Dimensions = 128 (8 groups × 16)

Requirements:
  pip install numpy pandas scikit-learn imbalanced-learn boruta psutil
  pip install tensorflow  # 2.10+ recommended
"""

import os, gc, json, time, random, warnings, shutil, subprocess, threading
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Conv1D, Dense, Multiply, Concatenate,
                                     LayerNormalization, GlobalAveragePooling1D,
                                     Reshape, Dropout, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# Sklearn & friends
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

# Imbalanced sampling
try:
    from imblearn.over_sampling import SMOTE
except Exception as e:
    raise SystemExit("Please install imbalanced-learn: pip install imbalanced-learn") from e

# Boruta
try:
    from boruta import BorutaPy
except Exception as e:
    raise SystemExit("Please install boruta: pip install boruta") from e

# System metrics
try:
    import psutil
except Exception as e:
    raise SystemExit("Please install psutil: pip install psutil") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# I/O config
# -----------------------------
DATA_PATH = r"C:/Users/shiva/Desktop/sabm_WITH_sCHEDULING/dfas.csv"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Utility: metrics
# -----------------------------
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

def metric_panel(y_true, y_prob, thr_list=THRESHOLDS):
    out = {}
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    for thr in thr_list:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        gmean = np.sqrt(max(0.0, tpr * tnr))
        mcc = matthews_corrcoef(y_true, y_pred) if (tp+tn+fp+fn) else 0.0
        out[thr] = dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1,
                        TPR=tpr, TNR=tnr, GMean=gmean, MCC=mcc, AUC=auc)
    return out

# Expected Calibration Error (ECE)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        idx = bin_ids == b
        if not np.any(idx):
            continue
        conf = y_prob[idx].mean()
        acc = y_true[idx].mean()
        w = idx.mean()
        ece += w * abs(acc - conf)
    return float(ece)

# -----------------------------
# Paper-style preprocessing + time features
# -----------------------------
STATUS_MAP = {"Terminated": 0, "Failed": 1}
DROP_STATUSES = {"Running", "Interrupted", "Waiting"}

def load_and_clean_df(path):
    df = pd.read_csv(path)

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # ------- Add 3 time-series features from 'start_date' -------
    # ts_hour, ts_dayofweek, ts_month
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce", utc=False)
        df["ts_hour"] = dt.dt.hour.fillna(0).astype(int)
        df["ts_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)   # 0=Mon,...,6=Sun
        df["ts_month"] = dt.dt.month.fillna(1).astype(int)           # 1..12
        # keep original 'start_date' out of features (non-numeric)
        # if you prefer to drop it entirely:
        # df = df.drop(columns=["start_date"])
    else:
        # If absent, create neutral placeholders so pipeline doesn't break
        df["ts_hour"] = 0
        df["ts_dayofweek"] = 0
        df["ts_month"] = 1

    # Normalize status columns -> Failed_Inst / Failed_Job (0/1)
    if "Failed_Inst" not in df.columns:
        if "status_i" in df.columns:
            df = df[~df["status_i"].isin(DROP_STATUSES)]
            df["Failed_Inst"] = df["status_i"].replace(STATUS_MAP)
        elif "Failed" in df.columns:
            df["Failed_Inst"] = df["Failed"].astype(int)
        else:
            raise ValueError("Neither Failed_Inst nor status_i/Failed columns found.")
    if "Failed_Job" not in df.columns:
        if "status_j" in df.columns:
            df = df[~df["status_j"].isin(DROP_STATUSES)]
            df["Failed_Job"] = df["status_j"].replace(STATUS_MAP)
        else:
            df["Failed_Job"] = df["Failed_Inst"].values

    # Make labels integer
    df["Failed_Inst"] = df["Failed_Inst"].astype(int)
    df["Failed_Job"] = df["Failed_Job"].astype(int)

    # Drop meta IDs/categoricals; keep numeric features
    non_feature_cols = {
        "job_name","task_name","inst_name","worker_name","inst_id","machine",
        "gpu_type","gpu_type_spec","group","user","gpu_name",
        "status","status_i","status_j","Failed","Failed_Inst","Failed_Job",
        "start_date"  # ensure this textual column is excluded from features
    }
    feat_df = df.drop(columns=[c for c in df.columns if c in non_feature_cols], errors="ignore")
    labels = df[["Failed_Inst","Failed_Job"]].copy()

    # Ensure numeric only
    feat_df = feat_df.select_dtypes(include=[np.number]).copy()

    # Fill missing with median (robust)
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))

    # MinMax scaling
    scaler = MinMaxScaler()
    feat_df_scaled = pd.DataFrame(scaler.fit_transform(feat_df), columns=feat_df.columns, index=feat_df.index)

    return feat_df_scaled, labels, scaler

# -----------------------------
# Feature selection: Boruta, MI, DFS (GB) — UNION (no cap)
# -----------------------------
def select_features_union_all(X, y, target_name, save_prefix):
    """
    Returns UNION of Boruta, MI(>median), DFS(top-10) — NO 24-feature cap.
    """
    # Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=SEED)
    boruta = BorutaPy(rf, n_estimators="auto", random_state=SEED, verbose=0)
    boruta.fit(X.values, y.values)
    boruta_feats = list(X.columns[boruta.support_])
    if len(boruta_feats) == 0:
        boruta_feats = list(X.columns[:min(5, X.shape[1])])

    # MI
    mi = mutual_info_classif(X.values, y.values, random_state=SEED)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    mi_thresh = mi_s.median()
    mi_feats = list(mi_s[mi_s > mi_thresh].index)
    if len(mi_feats) == 0:
        mi_feats = list(mi_s.head(min(10, len(mi_s))).index)

    # DFS via GB importance
    gb = GradientBoostingClassifier(random_state=SEED)
    gb.fit(X.values, y.values)
    imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
    dfs_feats = list(imp.head(min(10, len(imp))).index)

    # UNION (no cap)
    union = list(dict.fromkeys(boruta_feats + mi_feats + dfs_feats))

    # Save selection
    fs_record = {
        "target": target_name,
        "boruta": boruta_feats,
        "mi_above_median": mi_feats,
        "dfs_top10": dfs_feats,
        "final_union": union
    }
    with open(os.path.join(RESULTS_DIR, f"feature_selection_{save_prefix}_{target_name}.json"), "w") as f:
        json.dump(fs_record, f, indent=2)

    return union, fs_record

# -----------------------------
# System footprint monitoring
# -----------------------------
class FootprintMonitor:
    """
    Samples CPU%, RAM (RSS), and GPU util/mem via nvidia-smi if available.
    """
    def __init__(self, interval=0.5):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.stop_flag = threading.Event()
        self.thread = None
        self.cpu_readings = []
        self.rss_max = 0
        self.gpu_util_max = 0
        self.gpu_mem_max = 0
        self.has_nvsmi = shutil.which("nvidia-smi") is not None
        self.tf_gpu_peak_mb = 0

    def _sample_gpu_nvsmi(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip().splitlines()
            util = 0
            mem = 0
            for line in out:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    util = max(util, float(parts[0]))
                    mem = max(mem, float(parts[1]))
            return util, mem
        except Exception:
            return 0.0, 0.0

    def _sample_tf_gpu_mem(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                peak_mb = info.get('peak', 0) / (1024**2)
                return peak_mb
        except Exception:
            pass
        return 0.0

    def _run(self):
        self.proc.cpu_percent(interval=None)
        while not self.stop_flag.is_set():
            self.cpu_readings.append(self.proc.cpu_percent(interval=None))
            rss = self.proc.memory_info().rss
            self.rss_max = max(self.rss_max, rss)
            if self.has_nvsmi:
                util, mem = self._sample_gpu_nvsmi()
                self.gpu_util_max = max(self.gpu_util_max, util)
                self.gpu_mem_max = max(self.gpu_mem_max, mem)
            self.tf_gpu_peak_mb = max(self.tf_gpu_peak_mb, self._sample_tf_gpu_mem())
            time.sleep(self.interval)

    def start(self):
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join()
        cpu_avg = float(np.mean(self.cpu_readings)) if self.cpu_readings else 0.0
        cpu_max = float(np.max(self.cpu_readings)) if self.cpu_readings else 0.0
        return {
            "cpu_avg": round(cpu_avg, 2),
            "cpu_max": round(cpu_max, 2),
            "rss_max_mb": round(self.rss_max / (1024**2), 2),
            "gpu_util_max": round(self.gpu_util_max, 1),
            "gpu_mem_max_mb": round(self.gpu_mem_max, 1),
            "tf_gpu_peak_mb": round(self.tf_gpu_peak_mb, 1)
        }

# -----------------------------
# Custom layers & blocks
# -----------------------------
class DropBlock1D(keras.layers.Layer):
    def __init__(self, drop_prob=0.1, block_size=3, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def call(self, x, training=None):
        if (training is False) or self.drop_prob <= 0.0:
            return x
        if training is None:
            training = keras.backend.learning_phase()

        def dropped():
            b, t, c = tf.unstack(tf.shape(x))
            gamma = self.drop_prob * tf.cast(t, tf.float32) / tf.cast(tf.maximum(1, t - self.block_size + 1), tf.float32)
            M = tf.cast(tf.random.uniform((b, t, 1)) < gamma, tf.float32)
            M = tf.nn.max_pool1d(M, ksize=self.block_size, strides=1, padding='SAME')
            mask = 1.0 - M
            y = x * mask
            keep = tf.reduce_sum(mask) + 1e-6
            y = y * (tf.cast(tf.size(mask), tf.float32) / keep)
            return y

        return tf.cond(tf.cast(training, tf.bool), dropped, lambda: x)

class GroupedQueryAttention(keras.layers.Layer):
    def __init__(self, num_groups=4, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups

    def build(self, input_shape):
        self.time_steps = int(input_shape[1])
        self.features = int(input_shape[2])
        if self.features % self.num_groups != 0:
            raise ValueError(f"features ({self.features}) must be divisible by num_groups ({self.num_groups})")
        self.group_size = self.features // self.num_groups
        self.q_layers = [Dense(self.group_size) for _ in range(self.num_groups)]
        self.k_layers = [Dense(self.group_size) for _ in range(self.num_groups)]
        self.v_layers = [Dense(self.group_size) for _ in range(self.num_groups)]
        super().build(input_shape)

    def call(self, x):
        outs = []
        scale = tf.math.sqrt(tf.cast(self.group_size, tf.float32))
        for i in range(self.num_groups):
            xi = x[:, :, i*self.group_size:(i+1)*self.group_size]
            q = self.q_layers[i](xi)
            k = self.k_layers[i](xi)
            v = self.v_layers[i](xi)
            attn = tf.matmul(q, k, transpose_b=True) / (scale + 1e-9)
            attn = tf.nn.softmax(attn, axis=-1)
            outs.append(tf.matmul(attn, v))
        return tf.concat(outs, axis=-1)

def temporal_capsule_block(x, groups=8, dim=16):
    h = Conv1D(groups*dim, kernel_size=1, padding="same", activation="relu")(x)
    T = x.shape[1]
    G, D = groups, dim
    h = Reshape((T, G, D))(h)
    s2 = tf.reduce_sum(tf.square(h), axis=-1, keepdims=True)
    scale = s2 / (1.0 + s2)
    h = scale * h / tf.sqrt(s2 + keras.backend.epsilon())
    h = Reshape((T, G*D))(h)
    return h

def dual_attention_block(x):
    t_score = Dense(1, use_bias=False)(x)              # [B,T,1]
    alpha = tf.nn.softmax(t_score, axis=1)             # [B,T,1]
    attn_temp = tf.reduce_sum(alpha * x, axis=1)       # [B,F]
    g = tf.reduce_mean(x, axis=1)                      # [B,F]
    beta = tf.nn.sigmoid(Dense(x.shape[-1])(g))        # [B,F]
    attn_spat = beta * g                               # [B,F]
    fused = Concatenate(axis=-1)([attn_temp, attn_spat]) # [B,2F]
    return fused

def transformer_block(x, num_heads=4, ffn_hidden=256):
    attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1]//num_heads)(x, x)
    x = LayerNormalization()(x + attn)
    ff = Dense(ffn_hidden, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    x = LayerNormalization()(x + ff)
    return x

# -----------------------------
# GQAT-Net model
# -----------------------------
def build_gqat_net(input_shape, dropout=0.3, gqa_groups=4):
    inp = Input(shape=input_shape)          # [T, C] where T=#features as sequence, C=1

    # Multi-Scale Gated Convs across "feature sequence"
    conv1 = Conv1D(64, kernel_size=3, padding='same')(inp)
    gate1 = Conv1D(64, kernel_size=3, padding='same', activation='sigmoid')(inp)
    gc1 = Multiply()([conv1, gate1])

    conv2 = Conv1D(64, kernel_size=5, padding='same')(inp)
    gate2 = Conv1D(64, kernel_size=5, padding='same', activation='sigmoid')(inp)
    gc2 = Multiply()([conv2, gate2])

    conv3 = Conv1D(64, kernel_size=7, padding='same')(inp)
    gate3 = Conv1D(64, kernel_size=7, padding='same', activation='sigmoid')(inp)
    gc3 = Multiply()([conv3, gate3])

    x = Concatenate()([gc1, gc2, gc3])     # [T, 192]
    x = LayerNormalization()(x)

    # Temporal Capsules (128 dims = 8 groups × 16)
    x_caps = temporal_capsule_block(x, groups=8, dim=16)   # [T,128]

    # Dual attention on capsules (returns [B, 256]); Residual projection back to [T,128]
    fused = dual_attention_block(x_caps)     # [B,256]
    proj = Dense(128)(fused)                 # [B,128]
    proj_exp = Reshape((1, 128))(proj)
    proj_tiled = tf.tile(proj_exp, [1, tf.shape(x_caps)[1], 1])  # [B,T,128]
    x_res = Add()([x_caps, proj_tiled])      # [B,T,128]

    # Transformer encoder (1 block, 4 heads)
    x_trans = transformer_block(x_res, num_heads=4, ffn_hidden=256)  # [B,T,128]

    # Dynamic 1x1 conv modulated by global context
    gctx = GlobalAveragePooling1D()(x_trans)        # [B,128]
    dyn_w = Dense(64, activation='relu')(gctx)      # [B,64]
    x_dyn = Conv1D(64, 1, padding='same', activation='relu')(x_trans)  # [B,T,64]
    x_dyn = Multiply()([x_dyn, Reshape((1, 64))(dyn_w)])               # [B,T,64]

    # Adaptive DropBlock over feature sequence
    x_db = DropBlock1D(drop_prob=0.1, block_size=3)(x_dyn)

    # Grouped Query Attention over channels
    x_gqa = GroupedQueryAttention(num_groups=gqa_groups)(x_db)         # [B,T,64]

    # Global + Local fusion
    g_global = GlobalAveragePooling1D()(x_gqa)                         # [B,64]
    g_local = Dense(64, activation='relu')(x_gqa)                      # [B,T,64]
    g_local = GlobalAveragePooling1D()(g_local)                         # [B,64]

    feat = Concatenate()([g_global, g_local])                          # [B,128]
    feat = Dropout(dropout)(feat, training=True)  # keep active for MC-dropout at inference

    out = Dense(1, activation='sigmoid')(feat)
    model = Model(inputs=inp, outputs=out, name="GQAT_Net")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def count_params_and_size_mb(model):
    params = int(model.count_params())
    size_mb = round(params * 4 / (1024**2), 2)  # float32
    return params, size_mb

# -----------------------------
# MC-Dropout prediction
# -----------------------------
def mc_dropout_predict(model, X, n_iter=50, batch_size=256):
    preds = []
    for _ in range(n_iter):
        p = model(X, training=True).numpy().ravel()
        preds.append(p)
    P = np.stack(preds, axis=0)
    return P.mean(axis=0), P.std(axis=0)

# -----------------------------
# Main training / CV routine
# -----------------------------
def run_experiment():
    X_all, labels, scaler = load_and_clean_df(DATA_PATH)

    # Use the full dataset (no 20k sampling)
    sel_idx = X_all.index
    X_all = X_all.loc[sel_idx].reset_index(drop=True)
    labels = labels.loc[sel_idx].reset_index(drop=True)

    with open(os.path.join(RESULTS_DIR, "dataset_shape.json"), "w") as f:
        json.dump({"rows": int(len(X_all)), "features": int(X_all.shape[1])}, f, indent=2)

    results_all = []

    for target_col in ["Failed_Inst", "Failed_Job"]:
        print(f"\n=== Target: {target_col} ===")
        y = labels[target_col].values.astype(int)

        # Feature selection union (no cap)
        feats_union, fs_record = select_features_union_all(X_all, labels[target_col], target_col, save_prefix="dfas")
        X = X_all[feats_union].copy()

        # Prepare NN input: treat features as "sequence" length T, 1 channel
        def to_seq(arr):
            return arr.astype("float32").reshape((-1, arr.shape[1], 1))

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold = 1
        fold_logs = []

        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
            y_tr, y_te = y[tr_idx], y[te_idx]

            # train/val split
            X_trn, X_val, y_trn, y_val = train_test_split(
                X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED
            )

            # SMOTE on training
            sm = SMOTE(random_state=SEED)
            X_trn_sm, y_trn_sm = sm.fit_resample(X_trn, y_trn)

            # Build model (Dropout=0.3, LR=1e-4, Heads=4, Capsules=128=8x16)
            model = build_gqat_net(input_shape=(X.shape[1], 1), dropout=0.3, gqa_groups=4)
            params, size_mb = count_params_and_size_mb(model)

            # TRAIN with monitoring
            es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
            mon_train = FootprintMonitor(interval=0.5).start()
            t0 = time.time()
            history = model.fit(
                to_seq(X_trn_sm), y_trn_sm,
                validation_data=(to_seq(X_val), y_val),
                epochs=80, batch_size=128, callbacks=[es], verbose=0
            )
            t_train = time.time() - t0
            train_fp = mon_train.stop()

            # Calibration on val (fit isotonic on raw validation probs)
            p_val_raw = model.predict(to_seq(X_val), batch_size=256, verbose=0).ravel()
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_val_raw, y_val)

            # INFERENCE with monitoring
            mon_inf = FootprintMonitor(interval=0.5).start()
            t1 = time.time()
            p_te_raw = model.predict(to_seq(X_te), batch_size=256, verbose=0).ravel()
            t_inf = time.time() - t1
            inf_fp = mon_inf.stop()

            # Calibrated probs (apply isotonic to test probs)
            p_te_cal = iso.predict(p_te_raw)

            # Uncertainty via MC Dropout (mean probs)
            p_te_mc, p_te_std = mc_dropout_predict(model, to_seq(X_te), n_iter=50, batch_size=256)

            # Metrics (raw, calibrated, MC-mean) across multiple thresholds
            metr_raw = metric_panel(y_te, p_te_raw)
            metr_cal = metric_panel(y_te, p_te_cal)
            metr_mc  = metric_panel(y_te, p_te_mc)

            # ECE (raw vs recalibrated vs MC-mean)
            ece_raw = expected_calibration_error(y_te, p_te_raw, n_bins=10)
            ece_cal = expected_calibration_error(y_te, p_te_cal, n_bins=10)
            ece_mc  = expected_calibration_error(y_te, p_te_mc,  n_bins=10)

            # Curves (ROC/PR only) — NO calibration plots
            def save_curves(prefix, ytrue, prob):
                fpr, tpr, _ = roc_curve(ytrue, prob)
                plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(ytrue, prob):.3f}")
                plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
                plt.title(f"ROC — {prefix}")
                plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"ROC_{prefix}.png")); plt.close()

                prec, rec, _ = precision_recall_curve(ytrue, prob)
                plt.figure(); plt.plot(rec, prec)
                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {prefix}")
                plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"PR_{prefix}.png")); plt.close()

            prefix_base = f"{target_col}_fold{fold}"
            save_curves(prefix_base+"_raw", y_te, p_te_raw)
            save_curves(prefix_base+"_cal", y_te, p_te_cal)
            save_curves(prefix_base+"_mc",  y_te, p_te_mc)

            # Confusion matrix at 0.5 (raw)
            cm = confusion_matrix(y_te, (p_te_raw >= 0.5).astype(int))
            pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]).to_csv(
                os.path.join(RESULTS_DIR, f"CM_{prefix_base}_thr0.5_raw.csv"), index=True
            )

            # Save model graph & weights (first fold only for brevity)
            if fold == 1:
                model_path = os.path.join(RESULTS_DIR, f"gqat_{target_col}.h5")
                model.save(model_path)
                try:
                    plot_model(model, to_file=os.path.join(PLOTS_DIR, f"gqat_{target_col}_graph.png"),
                               show_shapes=True, show_layer_names=True)
                except Exception:
                    pass

            # Fold log with footprint + size/params + ECE
            fold_log = dict(
                target=target_col, fold=fold,
                params=params, size_mb=size_mb,
                hyperparams=dict(batch_size=128, learning_rate=1e-4, epochs=80,
                                 dropout=0.3, attention_heads=4, capsule_dims="128(8x16)"),
                train_time_sec=round(t_train,2), infer_time_sec=round(t_inf,2),
                train_footprint=train_fp, infer_footprint=inf_fp,
                ECE_raw=ece_raw, ECE_calibrated=ece_cal, ECE_mc=ece_mc,
                metrics_raw=metr_raw, metrics_calibrated=metr_cal, metrics_mc=metr_mc,
                feature_set=feats_union
            )
            with open(os.path.join(RESULTS_DIR, f"footprint_{prefix_base}.json"), "w") as f:
                json.dump(fold_log, f, indent=2)

            # Save uncertainty CSV
            unct_df = pd.DataFrame({"y_true": y_te, "p_raw": p_te_raw, "p_cal": p_te_cal,
                                    "p_mc": p_te_mc, "mc_std": p_te_std})
            unct_df.to_csv(os.path.join(RESULTS_DIR, f"uncertainty_{prefix_base}.csv"), index=False)

            # Cleanup
            del model, iso, history, X_tr, X_te, y_tr, y_te, X_trn, X_val, y_trn, y_val
            gc.collect()
            fold += 1

        with open(os.path.join(RESULTS_DIR, f"cv_summary_{target_col}.json"), "w") as f:
            json.dump(fold_logs, f, indent=2)

        results_all.extend(fold_logs)

    # Global summary
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(results_all, f, indent=2)

    # Compact CSV of F1@0.5 + ECE across folds
    rows = []
    for rec in results_all:
        f1_raw = rec["metrics_raw"][0.5]["F1"]
        f1_cal = rec["metrics_calibrated"][0.5]["F1"]
        f1_mc  = rec["metrics_mc"][0.5]["F1"]
        rows.append([
            rec["target"], rec["fold"], rec["params"], rec["size_mb"],
            rec["train_time_sec"], rec["infer_time_sec"],
            rec["train_footprint"]["cpu_avg"], rec["train_footprint"]["cpu_max"],
            rec["train_footprint"]["rss_max_mb"], rec["train_footprint"]["gpu_util_max"],
            rec["train_footprint"]["gpu_mem_max_mb"], rec["train_footprint"]["tf_gpu_peak_mb"],
            rec["infer_footprint"]["cpu_avg"], rec["infer_footprint"]["cpu_max"],
            rec["infer_footprint"]["rss_max_mb"], rec["infer_footprint"]["gpu_util_max"],
            rec["infer_footprint"]["gpu_mem_max_mb"], rec["infer_footprint"]["tf_gpu_peak_mb"],
            rec["ECE_raw"], rec["ECE_calibrated"], rec["ECE_mc"],
            f1_raw, f1_cal, f1_mc
        ])
    pd.DataFrame(rows, columns=[
        "Target","Fold","Params","ModelSizeMB","TrainTimeSec","InferTimeSec",
        "TrainCPUavg","TrainCPUmax","TrainRAMpeakMB","TrainGPUutilMax","TrainGPUmemMaxMB","TrainTFPeakMB",
        "InferCPUavg","InferCPUmax","InferRAMpeakMB","InferGPUutilMax","InferGPUmemMaxMB","InferTFPeakMB",
        "ECE_raw","ECE_calibrated","ECE_mc",
        "F1_raw@0.5","F1_cal@0.5","F1_mc@0.5"
    ]).to_csv(os.path.join(RESULTS_DIR, "summary_F1_ECE_footprint.csv"), index=False)

if __name__ == "__main__":
    run_experiment()
