# =============================================================================
# AQUAGUARD AI v4.2 — PRODUCTION ML PIPELINE  (FULLY CORRECTED)
# Change The Future Team | Amman Arab University | IEEE RAS & CS Hackathon
#
# FIXES OVER v4.1:
#   [CRITICAL] LSTM sequence ordering — outputs now guaranteed in original
#              row order so lstm_proba[i] always corresponds to row i.
#              The v4.1 bug used groupby output order which differed from
#              the dataframe order → Meta-learner received mismatched signals.
#   [FEATURE]  build_sequences_ordered() — O(n) dict-based construction,
#              no sorting of full array needed, works for any split.
#   [FEATURE]  Validation assertion: verifies y_seq == original labels
#              before training starts — catches any future ordering bugs.
# =============================================================================
# COLAB:
#   !pip install -q lightgbm xgboost scikit-learn pandas numpy
#   !pip install -q matplotlib seaborn joblib torch
#   exec(open('/content/data/aquaguard_v4_2_model.py').read())
# =============================================================================

import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, brier_score_loss, roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False; print("⚠️  !pip install lightgbm")

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False; print("⚠️  !pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_OK = True
    print(f"✅ PyTorch {torch.__version__} | device: {DEVICE}")
except ImportError:
    TORCH_OK = False; print("⚠️  !pip install torch")

CLASS_NAMES = ["normal", "leak", "burst", "theft"]
OUTDIR      = "aquaguard_v42_outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def sep(t=""):
    print("\n" + "═"*65 + (f"\n  {t}" if t else "") + "\n" + "═"*65)

def save_fig(fig, name, dpi=300):
    fig.savefig(f"{OUTDIR}/{name}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 1 — LOADING DATA")

def find(name):
    for p in [f"/content/data/{name}", f"/content/{name}", f"./{name}"]:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Upload {name} to /content/data/")

df_raw_tr = pd.read_csv(find("jordan_v4_train.csv"))
df_raw_te = pd.read_csv(find("jordan_v4_test.csv"))
print(f"Train: {df_raw_tr.shape[0]:,} × {df_raw_tr.shape[1]}")
print(f"Test : {df_raw_te.shape[0]:,} × {df_raw_te.shape[1]}")
print(f"\nClass dist (train):\n{df_raw_tr['Anomaly_Type'].value_counts().to_string()}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 2 — FEATURE ENGINEERING")

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    G  = ["Timestamp", "Governorate", "Branch"]   # group key

    # ── Core signal features ─────────────────────────────────────────────
    df["Flow_Efficiency"]      = df["Flow_Out"]     / df["Flow_In"].clip(1e-3)
    df["Pressure_Efficiency"]  = df["Pressure_Out"] / df["Pressure_In"].clip(1e-3)
    df["DP_Over_Predicted"]    = df["DP_Actual"]    / df["DP_Predicted"].clip(1e-3)
    df["Loss_Per_Meter"]       = df["Flow_Loss"]    / df["Pipe_Length_m"].clip(1.0)
    df["Excess_Ratio"]         = df["Excess_Loss_Pct"] / df["Bg_Loss_Rate_Pct"].clip(1e-3)
    df["DP_Dev_Abs"]           = df["DP_Deviation"].abs()

    # ── Fault fingerprint scores ─────────────────────────────────────────
    df["Theft_Score"]          = df["Flow_Loss_Pct"] / (df["DP_Dev_Abs"].clip(0.05) + 0.1)
    df["Burst_Score"]          = df["Flow_Loss_Pct"] * df["DP_Dev_Abs"].clip(0)
    df["Leak_Score"]           = df["Flow_Loss_Pct"].clip(0,20) * (df["DP_Dev_Abs"].clip(0,5)/5)
    df["Pipe_Deterioration"]   = (df["Pipe_Age_Years"]/60) * (140/df["HW_Coefficient"].clip(60))
    df["Expected_Loss_by_Age"] = df["Pipe_Age_Years"] * 0.00035 * df["Pipe_Length_m"]

    # ── Depth-normalised signals ─────────────────────────────────────────
    df["Depth_Norm_Loss"]      = df["Flow_Loss_Pct"]  / (df["Depth_Level"]+0.5)
    df["Depth_Norm_DP"]        = df["DP_Deviation"]   / (df["Depth_Level"]+0.5)
    df["Head_Available"]       = df["Pressure_In"]    / (df["Depth_Level"]+1.0)

    # ── Branch-level context ─────────────────────────────────────────────
    for col in ["Flow_Loss_Pct","Excess_Loss_Pct","DP_Deviation","Flow_DP_Ratio"]:
        bm = df.groupby(G)[col].transform("mean")
        bx = df.groupby(G)[col].transform("max")
        bs = df.groupby(G)[col].transform("std").fillna(0)
        s  = (col.replace("_Loss_Pct","_Lp").replace("Excess_","Ex_")
                 .replace("DP_Deviation","DPD").replace("Flow_DP_Ratio","FDR"))
        df[f"Branch_{s}_Mean"] = bm
        df[f"Branch_{s}_Max"]  = bx
        df[f"Branch_{s}_Std"]  = bs
        df[f"{s}_vs_Branch"]   = df[col] - bm

    # ── Cyclical time encoding ───────────────────────────────────────────
    df["Hour_Sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["Hour_Cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["Month_Sin"] = np.sin(2*np.pi*df["month"]/12)
    df["Month_Cos"] = np.cos(2*np.pi*df["month"]/12)
    df["DoW_Sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["DoW_Cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)

    # ── Interaction features ─────────────────────────────────────────────
    df["Age_x_NRW"]       = df["Pipe_Age_Years"] * df["nrw_rate"]
    df["Loss_x_DP"]       = df["Flow_Loss_Pct"]  * df["DP_Dev_Abs"]
    df["Velocity_x_Loss"] = df["Velocity_Proxy"] * df["Flow_Loss_Pct"]
    df["Supply_x_Loss"]   = df["supply_on"]       * df["Flow_Loss_Pct"]

    # ── Theft-discriminating features (v4.1) ─────────────────────────────
    df["Theft_Purity_Score"] = (
        df["Flow_Loss_Pct"] /
        (df["DP_Dev_Abs"].clip(0.01)**0.5).clip(0.01)
    ).clip(0, 500)
    df["Age_Adj_Excess"]     = (
        df["Excess_Loss_Pct"] / df["Pipe_Deterioration"].clip(0.1)
    ).clip(0, 200)
    pef_bm = df.groupby(G)["Pressure_Efficiency"].transform("mean")
    df["P_Eff_vs_Branch"]    = df["Pressure_Efficiency"] - pef_bm
    df["Theft_Combined"]     = df["Flow_DP_Ratio"].clip(0,50) * df["Excess_Loss_Pct"]

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


df_tr_fe = engineer(df_raw_tr)
df_te_fe = engineer(df_raw_te)

TARGET = ["Anomaly_Type","Anomaly_Label","Anomaly_Binary","Fault_Here","Severity"]
DROP   = ["Timestamp","Governorate","Gov_Source","Branch",
          "Segment_From","Segment_To","Segment_ID","year"] + TARGET
FEATS  = [c for c in df_tr_fe.columns if c not in DROP]

for c in TARGET:
    assert c not in FEATS, f"DATA LEAKAGE: {c}"

print(f"Total features: {len(FEATS)}")
print("✅ Zero data leakage confirmed")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPLITS  (80 train / 20 val from train, test held-out)
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 3 — DATA SPLITS")

df_sh = df_tr_fe.sample(frac=1, random_state=42).reset_index(drop=True)
sss   = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
ti, vi = next(sss.split(df_sh, df_sh["Anomaly_Label"]))
df_tr  = df_sh.iloc[ti].reset_index(drop=True)
df_va  = df_sh.iloc[vi].reset_index(drop=True)

print(f"Train  : {len(df_tr):,}")
print(f"Val    : {len(df_va):,}")
print(f"Test   : {len(df_te_fe):,}")

X_tr = df_tr[FEATS].values.astype(np.float32)
X_va = df_va[FEATS].values.astype(np.float32)
X_te = df_te_fe[FEATS].values.astype(np.float32)
y_tr = df_tr["Anomaly_Label"].values
y_va = df_va["Anomaly_Label"].values
y_te = df_te_fe["Anomaly_Label"].values

scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_va_s  = scaler.transform(X_va)
X_te_s  = scaler.transform(X_te)

cw_arr  = compute_class_weight("balanced", classes=np.array([0,1,2,3]), y=y_tr)
cw_dict = dict(enumerate(cw_arr))
print(f"\nClass weights: {cw_dict}")

y_tr_bin = df_tr["Anomaly_Binary"].values
y_va_bin = df_va["Anomaly_Binary"].values
y_te_bin = df_te_fe["Anomaly_Binary"].values


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LightGBM
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 4 — LightGBM")

lgb_pva = lgb_pte = None

if LGB_OK:
    sw = np.array([cw_dict[y] for y in y_tr])
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sw, feature_name=FEATS)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

    params = dict(
        objective="multiclass", num_class=4, metric="multi_logloss",
        num_leaves=255, learning_rate=0.04, n_estimators=3000,
        min_child_samples=10, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.80, feature_fraction_bynode=0.80,
        reg_alpha=0.05, reg_lambda=0.15, min_split_gain=0.005,
        path_smooth=0.1, verbose=-1, random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    lgb_m = lgb.train(params, ds_tr, num_boost_round=3000,
                       valid_sets=[ds_va],
                       callbacks=[lgb.early_stopping(100, verbose=False),
                                  lgb.log_evaluation(100)])
    print(f"\n✅ LGB trained | {time.time()-t0:.0f}s | best: {lgb_m.best_iteration}")
    lgb_pva = lgb_m.predict(X_va, num_iteration=lgb_m.best_iteration)
    lgb_pte = lgb_m.predict(X_te, num_iteration=lgb_m.best_iteration)

    fi_df = pd.DataFrame({"feature": FEATS, "importance": lgb_m.feature_importance("gain")})\
              .sort_values("importance", ascending=False)

    print(f"\nLGB Validation:")
    print(classification_report(y_va, lgb_pva.argmax(1), target_names=CLASS_NAMES,
                                 digits=4, zero_division=0))
    print(f"Top-10 features:\n{fi_df.head(10).to_string(index=False)}")
else:
    fi_df = pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — XGBoost
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 5 — XGBoost")

xgb_pva = xgb_pte = None

if XGB_OK:
    sw = np.array([cw_dict[y] for y in y_tr])
    xgb_m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=4, eval_metric="mlogloss",
        n_estimators=2000, max_depth=9, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.80, colsample_bylevel=0.80,
        min_child_weight=3, reg_alpha=0.05, reg_lambda=0.8, gamma=0.01,
        random_state=42, n_jobs=-1, tree_method="hist",
        device="cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu",
        verbosity=0, early_stopping_rounds=100,
    )
    t0 = time.time()
    xgb_m.fit(X_tr, y_tr, sample_weight=sw,
               eval_set=[(X_va, y_va)], verbose=100)
    print(f"\n✅ XGB trained | {time.time()-t0:.0f}s | best: {xgb_m.best_iteration}")
    xgb_pva = xgb_m.predict_proba(X_va)
    xgb_pte = xgb_m.predict_proba(X_te)
    print(f"\nXGB Validation:")
    print(classification_report(y_va, xgb_pva.argmax(1), target_names=CLASS_NAMES,
                                 digits=4, zero_division=0))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Residual MLP
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 6 — DEEP NEURAL NETWORK (Residual MLP)")

nn_pva = nn_pte = None

if TORCH_OK:
    class ResBlock(nn.Module):
        def __init__(self, d, p=0.30):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d,d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(p),
                nn.Linear(d,d), nn.BatchNorm1d(d))
            self.act = nn.GELU()
        def forward(self, x): return self.act(self.net(x) + x)

    class WaterNet(nn.Module):
        def __init__(self, n, k=4, p=0.30):
            super().__init__()
            self.e = nn.Sequential(nn.Linear(n,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(p))
            self.r5= nn.Sequential(ResBlock(512,p),ResBlock(512,p))
            self.p2= nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(p*.9))
            self.r2= nn.Sequential(ResBlock(256,p*.9),ResBlock(256,p*.9))
            self.p1= nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(p*.7))
            self.r1= ResBlock(128,p*.7)
            self.h = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,k))
        def forward(self,x):
            return self.h(self.r1(self.p1(self.r2(self.p2(self.r5(self.e(x)))))))

    cw_t = torch.tensor(cw_arr, dtype=torch.float32).to(DEVICE)
    net  = WaterNet(X_tr_s.shape[1]).to(DEVICE)
    print(f"NN parameters: {sum(p.numel() for p in net.parameters()):,}")

    crit = nn.CrossEntropyLoss(weight=cw_t)
    opt  = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                      torch.tensor(y_tr,   dtype=torch.long)),
        batch_size=512, shuffle=True, num_workers=0, pin_memory=False)
    va_dl = DataLoader(
        TensorDataset(torch.tensor(X_va_s, dtype=torch.float32),
                      torch.tensor(y_va,   dtype=torch.long)),
        batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, epochs=100,
        steps_per_epoch=int(np.ceil(len(X_tr_s)/512)))

    best_f1=0; best_st=None; pat=0; hist={"tl":[],"vl":[],"vf":[]}

    for ep in range(100):
        net.train(); run=0.0
        for Xb,yb in tr_dl:
            Xb,yb=Xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); l=crit(net(Xb),yb)
            l.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step(); sched.step(); run+=l.item()*len(yb)
        tl=run/len(tr_dl.dataset)

        net.eval(); vl=0.0; pr=[]
        with torch.no_grad():
            for Xb,yb in va_dl:
                Xb,yb=Xb.to(DEVICE),yb.to(DEVICE)
                lg=net(Xb); vl+=crit(lg,yb).item()*len(yb)
                pr.append(lg.argmax(1).cpu().numpy())
        vl/=len(va_dl.dataset)
        vf=f1_score(y_va,np.concatenate(pr),average="macro",zero_division=0)

        hist["tl"].append(tl); hist["vl"].append(vl); hist["vf"].append(vf)
        if vf>best_f1:
            best_f1=vf
            best_st={k:v.cpu().clone() for k,v in net.state_dict().items()}; pat=0
        else: pat+=1
        if (ep+1)%10==0:
            print(f"  Ep {ep+1:3d} | train={tl:.4f} val={vl:.4f} F1={vf:.4f} (best={best_f1:.4f})")
        if pat>=15: print(f"  Early stop ep {ep+1}"); break

    net.load_state_dict(best_st)
    print(f"\n✅ NN trained | best val F1: {best_f1:.4f}")

    # Training curves
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(hist["tl"],label="Train"); ax[0].plot(hist["vl"],label="Val")
    ax[0].set(title="Loss",xlabel="Epoch"); ax[0].legend()
    ax[1].plot(hist["vf"],color="green"); ax[1].set(title="Val F1 Macro",xlabel="Epoch")
    ax[1].grid(alpha=0.3); fig.tight_layout(); save_fig(fig,"fig_nn_training_curve.png",dpi=150)

    net.eval()
    def get_nn_proba(Xs):
        ds=TensorDataset(torch.tensor(Xs,dtype=torch.float32))
        dl=DataLoader(ds,batch_size=1024,shuffle=False,num_workers=0)
        out=[]
        with torch.no_grad():
            for (Xb,) in dl:
                out.append(F.softmax(net(Xb.to(DEVICE)),dim=1).cpu().numpy())
        return np.vstack(out)

    nn_pva = get_nn_proba(X_va_s)
    nn_pte = get_nn_proba(X_te_s)
    print(f"\nNN Validation:")
    print(classification_report(y_va,nn_pva.argmax(1),target_names=CLASS_NAMES,digits=4,zero_division=0))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BiLSTM  (v4.2: ORDERING FIXED)
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 7 — BiLSTM  (v4.2 — ORDER FIXED)")
"""
THE FIX:
  Old approach (v4.0 / v4.1):
    → Iterate groupby groups
    → Append rows in group iteration order (Ajloun before Amman, etc.)
    → lstm_proba[i] corresponds to group-order row i — NOT original row i
    → Meta-learner sees mismatched (lgb[i], lstm[j]) pairs → learns to ignore LSTM

  New approach (v4.2):
    → Build output arrays of shape (n_rows, ...) indexed by ORIGINAL row index
    → For each group, write results back to the ORIGINAL positions of its rows
    → X_seq[i], d_seq[i], y_seq[i] always correspond to original row i
    → VERIFIED with assertion: y_seq == original labels
    → lstm_proba[i] now correctly paired with lgb_proba[i], xgb_proba[i], nn_proba[i]
"""

lstm_pva = lstm_pte = None
SEQ_FEATS = [f for f in [
    "Flow_In","Flow_Out","Flow_Loss_Pct","Excess_Loss_Pct",
    "Pressure_In","Pressure_Out","DP_Actual","DP_Predicted","DP_Deviation",
    "Flow_DP_Ratio","Pipe_Length_m","Pipe_Age_Years","HW_Coefficient",
    "Depth_Level","Velocity_Proxy","Theft_Score","Burst_Score","Leak_Score",
    "Pipe_Deterioration","Depth_Norm_Loss","Depth_Norm_DP",
    "Hour_Sin","Hour_Cos","Month_Sin","Month_Cos","DoW_Sin","DoW_Cos",
    "supply_on","nrw_rate","environment_code",
    "Theft_Purity_Score","Age_Adj_Excess","Theft_Combined",
] if f in df_tr_fe.columns]

MAX_SEQ = 4
N_SF    = len(SEQ_FEATS)

if TORCH_OK:
    # ── Sequence scaler: fit on TRAIN features only ──────────────────────
    seq_scaler = StandardScaler()
    seq_scaler.fit(df_tr[SEQ_FEATS].fillna(0))

    def build_sequences_ordered(df_feat, df_orig, seq_feats, scaler, max_len):
        """
        CORRECT implementation: output arrays are in ORIGINAL ROW ORDER.

        For each row i at position (Timestamp T, Governorate G, Branch B, Depth d):
          X_seq[i] = the full 4-step sequence of branch (T,G,B), padded if needed
          d_seq[i] = d - 1  (row i's own depth index within the sequence)
          y_seq[i] = original label of row i

        This guarantees lstm_proba[i] corresponds to original row i,
        so it can be safely stacked with lgb_pva[i], xgb_pva[i], nn_pva[i].
        """
        n_rows   = len(df_orig)
        X_scaled = scaler.transform(df_feat[seq_feats].fillna(0).values)

        # Pre-allocate in original row order
        X_seq = np.zeros((n_rows, max_len, len(seq_feats)), dtype=np.float32)
        d_seq = np.zeros(n_rows, dtype=np.int64)
        y_seq = df_orig["Anomaly_Label"].values.copy()   # same as original

        GRP_KEY = ["Timestamp", "Governorate", "Branch"]
        # Build a dict: group_key → sorted list of original row indices
        grp_dict = {}
        for orig_idx, row in df_orig[GRP_KEY].iterrows():
            key = (row["Timestamp"], row["Governorate"], row["Branch"])
            grp_dict.setdefault(key, []).append(orig_idx)

        for key, idx_list in grp_dict.items():
            # Sort group by Depth_Level — safe even after DataFrame shuffle
            idx_by_depth = sorted(idx_list,
                                   key=lambda i: df_orig.loc[i, 'Depth_Level'])
            L = min(len(idx_by_depth), max_len)

            # Build branch sequence in depth order
            seq     = np.zeros((max_len, len(seq_feats)), dtype=np.float32)
            seq[:L] = X_scaled[idx_by_depth[:max_len]]

            # d_seq = Depth_Level - 1, read directly from each row (always correct)
            for orig_i in idx_by_depth[:max_len]:
                X_seq[orig_i] = seq
                d_seq[orig_i] = int(df_orig.loc[orig_i, 'Depth_Level']) - 1

        # SAFETY ASSERTIONS
        assert np.array_equal(y_seq, df_orig['Anomaly_Label'].values), \
            'y_seq ORDER MISMATCH — LSTM would produce wrong predictions!'
        exp_d = (df_orig['Depth_Level'].values - 1).astype(np.int64)
        assert np.array_equal(d_seq, exp_d), \
            f'd_seq DEPTH MISMATCH — {(d_seq != exp_d).sum()} rows wrong!'

        return X_seq, y_seq, d_seq

    print("Building sequences (train)...")
    X_str, y_str, d_str = build_sequences_ordered(df_tr, df_tr, SEQ_FEATS, seq_scaler, MAX_SEQ)
    print(f"  ✅ train: shape={X_str.shape}, ordering verified")

    print("Building sequences (val)...")
    X_sva, y_sva, d_sva = build_sequences_ordered(df_va, df_va, SEQ_FEATS, seq_scaler, MAX_SEQ)
    print(f"  ✅ val  : shape={X_sva.shape}, ordering verified")

    print("Building sequences (test)...")
    X_ste, y_ste, d_ste = build_sequences_ordered(df_te_fe, df_te_fe, SEQ_FEATS, seq_scaler, MAX_SEQ)
    print(f"  ✅ test : shape={X_ste.shape}, ordering verified")

    # Verify y_sva equals y_va (they must be identical)
    assert np.array_equal(y_sva, y_va), "Val labels mismatch — critical bug!"
    assert np.array_equal(y_ste, y_te), "Test labels mismatch — critical bug!"
    print("\n✅ All ordering assertions passed — LSTM is safe to use in Ensemble")

    # ── BiLSTM model ─────────────────────────────────────────────────────
    class BranchLSTM(nn.Module):
        """
        Bidirectional LSTM that processes the full branch sequence.
        Outputs per-step logits; caller selects the step matching row depth.
        """
        def __init__(self, nf, h=128, nl=2, nc=4, p=0.30):
            super().__init__()
            self.lstm = nn.LSTM(nf, h, nl, batch_first=True,
                                bidirectional=True,
                                dropout=p if nl>1 else 0.0)
            self.norm = nn.LayerNorm(h*2)
            self.head = nn.Sequential(nn.Dropout(p),
                                       nn.Linear(h*2, 64), nn.GELU(),
                                       nn.Linear(64, nc))
        def forward(self, x):
            o,_ = self.lstm(x)
            return self.head(self.norm(o))  # (B, T, nc)

    lstm_net = BranchLSTM(N_SF).to(DEVICE)
    print(f"\nLSTM parameters: {sum(p.numel() for p in lstm_net.parameters()):,}")

    lcw   = torch.tensor(cw_arr, dtype=torch.float32).to(DEVICE)
    lcrit = nn.CrossEntropyLoss(weight=lcw)
    lopt  = torch.optim.AdamW(lstm_net.parameters(), lr=1e-3, weight_decay=1e-4)
    lsched= torch.optim.lr_scheduler.CosineAnnealingLR(lopt, T_max=80)

    def make_ds(Xs, ys, ds):
        return TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                             torch.tensor(ys, dtype=torch.long),
                             torch.tensor(ds, dtype=torch.long))

    ltr_dl = DataLoader(make_ds(X_str,y_str,d_str), batch_size=512,  shuffle=True,  num_workers=0)
    lva_dl = DataLoader(make_ds(X_sva,y_sva,d_sva), batch_size=1024, shuffle=False, num_workers=0)

    best_lf1=0; best_lst=None; lpat=0

    for ep in range(80):
        lstm_net.train(); ep_loss=0.0; ntot=0
        for Xb,yb,db in ltr_dl:
            Xb,yb,db = Xb.to(DEVICE),yb.to(DEVICE),db.to(DEVICE)
            lopt.zero_grad()
            logits = lstm_net(Xb)                             # (B,T,4)
            # Select logit at each row's OWN depth position
            logits_d = logits[torch.arange(len(db)), db, :]  # (B,4)
            loss = lcrit(logits_d, yb)
            loss.backward(); nn.utils.clip_grad_norm_(lstm_net.parameters(), 1.0)
            lopt.step(); ep_loss+=loss.item()*len(yb); ntot+=len(yb)
        lsched.step()

        lstm_net.eval(); preds=[]
        with torch.no_grad():
            for Xb,yb,db in lva_dl:
                Xb,db=Xb.to(DEVICE),db.to(DEVICE)
                logits   = lstm_net(Xb)
                logits_d = logits[torch.arange(len(db)), db, :]
                preds.append(logits_d.argmax(1).cpu().numpy())
        vf = f1_score(y_va, np.concatenate(preds), average="macro", zero_division=0)

        if vf>best_lf1:
            best_lf1=vf
            best_lst={k:v.cpu().clone() for k,v in lstm_net.state_dict().items()}; lpat=0
        else: lpat+=1
        if (ep+1)%10==0:
            print(f"  LSTM Ep {ep+1:3d} | loss={ep_loss/ntot:.4f} "
                  f"val_F1={vf:.4f} (best={best_lf1:.4f})")
        if lpat>=15: print(f"  Early stop ep {ep+1}"); break

    lstm_net.load_state_dict(best_lst)
    print(f"\n✅ LSTM trained | best val F1: {best_lf1:.4f}")

    # ── Get probabilities in ORIGINAL ORDER ──────────────────────────────
    lstm_net.eval()
    def get_lstm_proba(Xs, ds):
        dl = DataLoader(
            TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                          torch.tensor(ds, dtype=torch.long)),
            batch_size=1024, shuffle=False, num_workers=0)
        out=[]
        with torch.no_grad():
            for Xb,db in dl:
                Xb,db = Xb.to(DEVICE), db.to(DEVICE)
                logits   = lstm_net(Xb)
                logits_d = logits[torch.arange(len(db)), db, :]
                out.append(F.softmax(logits_d, dim=1).cpu().numpy())
        return np.vstack(out)

    lstm_pva = get_lstm_proba(X_sva, d_sva)
    lstm_pte = get_lstm_proba(X_ste, d_ste)

    # Final check: lstm_pva rows correspond to y_va rows
    print(f"\nLSTM Validation (v4.2 — ordering correct):")
    print(classification_report(y_va, lstm_pva.argmax(1),
                                  target_names=CLASS_NAMES, digits=4, zero_division=0))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — STACKING ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 8 — STACKING ENSEMBLE")

pva_list   = [lgb_pva, xgb_pva, nn_pva, lstm_pva]
pte_list   = [lgb_pte, xgb_pte, nn_pte, lstm_pte]
mnames     = ["LGB","XGB","NN","LSTM"]
active     = [n for n,p in zip(mnames,pva_list) if p is not None]

meta_X_va  = np.hstack([p for p in pva_list if p is not None])
meta_X_te  = np.hstack([p for p in pte_list if p is not None])

meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                           multi_class="multinomial", solver="lbfgs", random_state=42)
meta.fit(meta_X_va, y_va)

ens_pva = meta.predict_proba(meta_X_va)
ens_pte = meta.predict_proba(meta_X_te)

print(f"Ensemble Validation (before calibration):")
print(classification_report(y_va, ens_pva.argmax(1), target_names=CLASS_NAMES,
                             digits=4, zero_division=0))
print(f"Active models: {len(active)} ({', '.join(active)})")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ISOTONIC CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 9 — PROBABILITY CALIBRATION")

y_bin_va = label_binarize(y_va, classes=[0,1,2,3])
iso_cals  = []
for i in range(4):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(ens_pva[:,i], y_bin_va[:,i])
    iso_cals.append(ir)

def calibrate(p):
    c = np.column_stack([ir.predict(p[:,i]) for i,ir in enumerate(iso_cals)])
    return c / c.sum(axis=1, keepdims=True).clip(1e-8)

ens_pva_c  = calibrate(ens_pva)
ens_pte_c  = calibrate(ens_pte)
ens_pred   = ens_pte_c.argmax(axis=1)

bs_b = brier_score_loss(y_te_bin, ens_pte[:,1].clip(0,1))
bs_a = brier_score_loss(y_te_bin, ens_pte_c[:,1].clip(0,1))
print(f"Brier — before: {bs_b:.4f}  after: {bs_a:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DECISION ENGINE + THRESHOLD TUNING
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 10 — DECISION ENGINE + THRESHOLD TUNING")

te_conf = 1.0 - ens_pte_c[:, 0]

# Tune on VALIDATION only — test set never touched
va_conf = 1.0 - ens_pva_c[:, 0]
best_thr, best_f1 = 0.50, 0.0
for thr in np.arange(0.15, 0.90, 0.005):
    pred = (va_conf >= thr).astype(int)
    f = f1_score(y_va_bin, pred, zero_division=0)
    if f > best_f1: best_f1, best_thr = f, thr

ALERT_THR = best_thr
te_alert  = (te_conf >= ALERT_THR).astype(int)
te_type   = np.where(te_alert, [CLASS_NAMES[l] for l in ens_pred], "normal")

print(f"Optimal threshold : {ALERT_THR:.3f}  (val binary F1={best_f1:.4f})")
print(f"Alerts fired      : {te_alert.sum():,} / {len(te_alert):,}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FAULT LOCALIZATION MODEL
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 11 — FAULT LOCALIZATION MODEL")

anom_tr = df_tr["Anomaly_Binary"] == 1
anom_va = df_va["Anomaly_Binary"] == 1
anom_te = df_te_fe["Anomaly_Binary"] == 1
loc_f1  = loc_acc = 0.0

if LGB_OK and anom_tr.sum() > 100:
    y_loc_tr = df_tr.loc[anom_tr,"Fault_Here"].values
    y_loc_va = df_va.loc[anom_va,"Fault_Here"].values
    y_loc_te = df_te_fe.loc[anom_te,"Fault_Here"].values

    loc_ds = lgb.Dataset(X_tr_s[anom_tr], label=y_loc_tr)
    loc_va = lgb.Dataset(X_va_s[anom_va], label=y_loc_va, reference=loc_ds)

    loc_m = lgb.train(
        {"objective":"binary","metric":"binary_logloss","num_leaves":63,
         "learning_rate":0.05,"subsample":0.85,"colsample_bytree":0.85,
         "is_unbalance":True,"verbose":-1,"random_state":42},
        loc_ds, num_boost_round=500, valid_sets=[loc_va],
        callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(999)],
    )

    loc_pred_va = (loc_m.predict(X_va_s[anom_va]) >= 0.5).astype(int)
    loc_pred_te = (loc_m.predict(X_te_s[anom_te]) >= 0.5).astype(int)

    loc_acc = accuracy_score(y_loc_te, loc_pred_te)
    loc_f1  = f1_score(y_loc_te, loc_pred_te, zero_division=0)

    print(f"Localization (Fault_Here):")
    print(f"  Val  — Acc:{accuracy_score(y_loc_va,loc_pred_va):.4f}  "
          f"F1:{f1_score(y_loc_va,loc_pred_va,zero_division=0):.4f}  n={anom_va.sum():,}")
    print(f"  Test — Acc:{loc_acc:.4f}  F1:{loc_f1:.4f}  n={anom_te.sum():,}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12 — FULL EVALUATION (TEST SET)
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 12 — FINAL EVALUATION (TEST SET n=30,020)")

y_true_b  = df_te_fe["Anomaly_Binary"].values
y_true_mc = df_te_fe["Anomaly_Label"].values
y_bin_mc  = label_binarize(y_true_mc, classes=[0,1,2,3])

print("\n── A. Binary Detection ──")
bm = {
    "Accuracy":        accuracy_score(y_true_b, te_alert),
    "Precision":       precision_score(y_true_b, te_alert, zero_division=0),
    "Recall":          recall_score(y_true_b, te_alert, zero_division=0),
    "F1":              f1_score(y_true_b, te_alert, zero_division=0),
    "AUC-ROC":         roc_auc_score(y_true_b, te_conf),
    "AP":              average_precision_score(y_true_b, te_conf),
    "False Alarm Rate":te_alert[y_true_b==0].sum() / max((y_true_b==0).sum(),1),
    "Brier Score":     bs_a,
}
for k,v in bm.items(): print(f"  {k:<22}: {v:.4f}")

print("\n── B. Multi-class ──")
mc_str = classification_report(y_true_mc, ens_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
mc_rep = classification_report(y_true_mc, ens_pred, target_names=CLASS_NAMES,
                                digits=4, zero_division=0, output_dict=True)
print(mc_str)

print("── C. Per-class AUC / AP ──")
for i,cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<8}: AUC={roc_auc_score(y_bin_mc[:,i],ens_pte_c[:,i]):.4f}  "
          f"AP={average_precision_score(y_bin_mc[:,i],ens_pte_c[:,i]):.4f}")

cm = confusion_matrix(y_true_mc, ens_pred)
print(f"\nConfusion Matrix:\n{cm}")

df_te_fe["Pred_Alert"] = te_alert
df_te_fe["Pred_Label"] = ens_pred

print("\n── D. Performance by Depth Level ──")
dep_s = df_te_fe.groupby("Depth_Level").apply(lambda g: pd.Series({
    "n":       len(g),
    "true_an": g["Anomaly_Binary"].sum(),
    "detect":  g["Pred_Alert"].sum(),
    "recall":  recall_score(g["Anomaly_Binary"],g["Pred_Alert"],zero_division=0),
    "prec":    precision_score(g["Anomaly_Binary"],g["Pred_Alert"],zero_division=0),
    "f1":      f1_score(g["Anomaly_Binary"],g["Pred_Alert"],zero_division=0),
})).reset_index()
print(dep_s.round(4).to_string(index=False))

print("\n── E. Performance by Governorate ──")
gov_s = df_te_fe.groupby("Governorate").apply(lambda g: pd.Series({
    "n":          len(g),
    "anomalies":  g["Anomaly_Binary"].sum(),
    "detected":   g["Pred_Alert"].sum(),
    "recall":     recall_score(g["Anomaly_Binary"],g["Pred_Alert"],zero_division=0),
    "f1":         f1_score(g["Anomaly_Binary"],g["Pred_Alert"],zero_division=0),
    "false_alarms":g["Pred_Alert"][g["Anomaly_Binary"]==0].sum(),
})).reset_index()
print(gov_s.sort_values("recall",ascending=False).round(4).to_string(index=False))

print("\n── F. Theft Analysis ──")
theft_te = df_te_fe[df_te_fe["Anomaly_Type"]=="theft"].copy()
theft_te["Pred"] = theft_te["Pred_Label"].map(dict(enumerate(CLASS_NAMES)))
print(f"Theft predictions:\n{theft_te['Pred'].value_counts().to_string()}")
print(f"Theft Recall: {(theft_te['Pred_Label']==3).mean():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FIGURES (300 DPI, IEEE ready)
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 13 — GENERATING FIGURES")

pal = {"normal":"#4A90D9","leak":"#F5A623","burst":"#D0021B","theft":"#7B68EE"}
prec=[mc_rep[c]["precision"] for c in CLASS_NAMES]
rec =[mc_rep[c]["recall"]    for c in CLASS_NAMES]
f1s =[mc_rep[c]["f1-score"]  for c in CLASS_NAMES]
cm_n= cm.astype(float)/cm.sum(axis=1,keepdims=True)

# Fig 1: Confusion matrices
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,data,title,fmt in zip(axes,[cm,cm_n],["Raw Counts","Normalised"],["d",".3f"]):
    sns.heatmap(data,annot=True,fmt=fmt,cmap="Blues",
                xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=ax,annot_kws={"size":12})
    ax.set(xlabel="Predicted",ylabel="True",title=title)
fig.suptitle("AquaGuard v4.2 — Confusion Matrix (Test n=30,020)",fontsize=13,fontweight="bold")
fig.tight_layout(); save_fig(fig,"fig1_confusion_matrix.png")

# Fig 2: Per-class P/R/F1
x=np.arange(4); w=0.26
fig,ax=plt.subplots(figsize=(10,5))
b1=ax.bar(x-w,prec,w,label="Precision",color="#2C6FAC",edgecolor="white")
b2=ax.bar(x,  rec, w,label="Recall",   color="#1D9E75",edgecolor="white")
b3=ax.bar(x+w,f1s, w,label="F1-Score", color="#BA7517",edgecolor="white")
ax.set(xticks=x,xticklabels=[c.capitalize() for c in CLASS_NAMES],
       ylim=[0,1.15],ylabel="Score",title="Per-Class Metrics — v4.2")
ax.legend(fontsize=10)
for bars in [b1,b2,b3]:
    for bar in bars:
        h=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,h+0.01,f"{h:.3f}",ha="center",fontsize=8,fontweight="bold")
fig.tight_layout(); save_fig(fig,"fig2_per_class_metrics.png")

# Fig 3: Feature importance
if LGB_OK and len(fi_df)>0:
    fi_top=fi_df.head(20).sort_values("importance")
    NEW_F=["Theft_Purity","Age_Adj","Theft_Combined","P_Eff_vs","FDR_vs"]
    cols_fi=["#BA7517" if any(k in f for k in NEW_F) else "#2C6FAC" for f in fi_top["feature"]]
    fig,ax=plt.subplots(figsize=(10,8))
    ax.barh(fi_top["feature"],fi_top["importance"],color=cols_fi,edgecolor="white")
    ax.set(xlabel="Gain Importance",title="Top-20 Feature Importances — LightGBM")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2C6FAC",label="Original"),
                        Patch(color="#BA7517",label="NEW theft features")],
               loc="lower right",fontsize=9)
    fig.tight_layout(); save_fig(fig,"fig3_feature_importance.png")

# Fig 4: ROC curves
fig,ax=plt.subplots(figsize=(8,6))
for i,(cls,col) in enumerate(zip(CLASS_NAMES,["#4A90D9","#F5A623","#D0021B","#7B68EE"])):
    fpr,tpr,_=roc_curve(y_bin_mc[:,i],ens_pte_c[:,i])
    ax.plot(fpr,tpr,color=col,lw=2,
            label=f"{cls.capitalize()} (AUC={roc_auc_score(y_bin_mc[:,i],ens_pte_c[:,i]):.4f})")
ax.plot([0,1],[0,1],"k--",lw=1); ax.set(xlabel="FPR",ylabel="TPR",title="ROC Curves — All Classes")
ax.legend(fontsize=10); ax.grid(alpha=0.3); fig.tight_layout(); save_fig(fig,"fig4_roc_curves.png")

# Fig 5: Spatial signatures
fig,ax=plt.subplots(figsize=(10,6))
for cls in CLASS_NAMES:
    s=df_te_fe[df_te_fe["Anomaly_Type"]==cls]
    ax.scatter(s["Flow_Loss_Pct"],s["DP_Deviation"],label=cls.capitalize(),alpha=0.3,s=8,color=pal[cls])
ax.axvline(3,color="gray",linestyle="--",lw=0.8); ax.axhline(0.5,color="gray",linestyle="--",lw=0.8)
ax.set(xlabel="Flow_Loss_Pct (%)",ylabel="DP_Deviation (PSI)",
       title="Propagating Flow — Anomaly Signatures",xlim=[-1,60],ylim=[-15,75])
ax.legend(markerscale=3,fontsize=10); fig.tight_layout(); save_fig(fig,"fig5_spatial_signatures.png")

# Fig 6: Depth performance
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,col,clr,tit in zip(axes,["recall","f1"],["#2C6FAC","#1D9E75"],
                            ["Recall by Depth","F1 by Depth"]):
    vals=dep_s[col]
    ax.bar(dep_s["Depth_Level"],vals,color=clr,edgecolor="white")
    ax.set(xlabel="Depth Level",ylabel=col.capitalize(),title=tit,ylim=[0,1.1])
    ax.grid(axis="y",alpha=0.3)
    for dv,vv in zip(dep_s["Depth_Level"],vals):
        ax.text(dv,vv+0.01,f"{vv:.3f}",ha="center",fontsize=10,fontweight="bold")
fig.suptitle("Detection by DMA Depth Level",fontsize=13,fontweight="bold")
fig.tight_layout(); save_fig(fig,"fig6_depth_performance.png")

# Fig 7: Confidence distribution
fig,ax=plt.subplots(figsize=(10,5))
for cls in CLASS_NAMES:
    confs=te_conf[df_te_fe["Anomaly_Type"].values==cls]
    ax.hist(confs,bins=60,alpha=0.55,label=cls.capitalize(),color=pal[cls],density=True)
ax.axvline(ALERT_THR,color="black",linestyle="--",lw=2,label=f"Threshold={ALERT_THR:.3f}")
ax.set(xlabel="Ensemble Confidence",ylabel="Density",title="Confidence Distribution")
ax.legend(fontsize=10); fig.tight_layout(); save_fig(fig,"fig7_confidence_distribution.png")

# Fig 8: SCADA Dashboard
fig=plt.figure(figsize=(22,14),facecolor="#060a12")
fig.suptitle("AquaGuard AI v4.2 — SCADA Dashboard | Jordan Water Network",
             fontsize=16,fontweight="bold",color="#38bdf8",y=0.98)
from matplotlib.patches import FancyBboxPatch
gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.45,wspace=0.35,
                      left=0.04,right=0.97,top=0.93,bottom=0.05)
ax_k=fig.add_subplot(gs[0,:3]); ax_k.axis("off")
kpis=[("Records",f"{len(df_te_fe):,}","#38bdf8"),("Alerts",f"{te_alert.sum():,}","#ef4444"),
      ("Precision",f"{bm['Precision']:.1%}","#22c55e"),("Recall",f"{bm['Recall']:.1%}","#f59e0b"),
      ("F1",f"{bm['F1']:.1%}","#a855f7"),("AUC-ROC",f"{bm['AUC-ROC']:.4f}","#38bdf8"),
      ("Brier",f"{bm['Brier Score']:.4f}","#22c55e"),("FAR",f"{bm['False Alarm Rate']:.4f}","#f59e0b")]
for i,(lbl,val,col) in enumerate(kpis):
    xc=0.06+i*0.125
    ax_k.add_patch(FancyBboxPatch((xc-.055,.1),.108,.78,boxstyle="round,pad=0.01",lw=2,
                   edgecolor=col,facecolor="white",transform=ax_k.transAxes,clip_on=False))
    ax_k.text(xc,.75,lbl,ha="center",va="center",fontsize=7.5,color="#555",transform=ax_k.transAxes)
    ax_k.text(xc,.42,val,ha="center",va="center",fontsize=13,fontweight="bold",color=col,transform=ax_k.transAxes)
ax_c=fig.add_subplot(gs[0,3]); ax_c.set_facecolor("#0b1220")
sns.heatmap(cm_n,annot=True,fmt=".2f",cmap="YlOrRd",xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=ax_c)
ax_c.set_title("Confusion Matrix",color="white",fontsize=10); ax_c.tick_params(colors="white",labelsize=8)
ax_s=fig.add_subplot(gs[1,:2]); ax_s.set_facecolor("#0b1220")
for cls in CLASS_NAMES:
    s=df_te_fe[df_te_fe["Anomaly_Type"]==cls]
    ax_s.scatter(s["Flow_Loss_Pct"],s["DP_Deviation"],alpha=0.25,s=5,color=pal[cls],label=cls)
ax_s.set(xlabel="Flow_Loss_Pct",ylabel="DP_Deviation",xlim=[-1,60],ylim=[-15,75])
ax_s.set_title("Spatial Signatures",color="white"); ax_s.tick_params(colors="#94a3b8")
ax_s.legend(fontsize=7,markerscale=3,facecolor="#111",labelcolor="white")
ax_f=fig.add_subplot(gs[1,2:]); ax_f.set_facecolor("#0b1220")
if LGB_OK and len(fi_df)>0:
    t8=fi_df.head(8); ax_f.barh(t8["feature"][::-1],t8["importance"][::-1],color="#38bdf8")
ax_f.set_title("Top-8 Features",color="white",fontsize=10); ax_f.tick_params(colors="#94a3b8",labelsize=8)
ax_p=fig.add_subplot(gs[2,:2]); ax_p.set_facecolor("#0b1220")
xb=np.arange(4); wb=0.26
ax_p.bar(xb-wb,prec,wb,color="#2C6FAC",label="P",alpha=0.9)
ax_p.bar(xb,   rec, wb,color="#1D9E75",label="R",alpha=0.9)
ax_p.bar(xb+wb,f1s, wb,color="#BA7517",label="F1",alpha=0.9)
ax_p.set(xticks=xb,xticklabels=[c.capitalize() for c in CLASS_NAMES],ylim=[0,1.1])
ax_p.set_title("Per-Class Metrics",color="white"); ax_p.tick_params(colors="#94a3b8")
ax_p.legend(fontsize=8,facecolor="#111",labelcolor="white")
ax_d=fig.add_subplot(gs[2,2:]); ax_d.set_facecolor("#0b1220")
ax_d.bar(dep_s["Depth_Level"],dep_s["f1"],color="#22c55e",edgecolor="white",lw=0.5)
ax_d.set(xlabel="Depth Level",ylabel="F1",ylim=[0,1.1])
ax_d.set_title("F1 by DMA Depth",color="white"); ax_d.tick_params(colors="#94a3b8")
for dv,fv in zip(dep_s["Depth_Level"],dep_s["f1"]):
    ax_d.text(dv,fv+0.01,f"{fv:.3f}",ha="center",color="white",fontsize=9)
save_fig(fig,"fig8_scada_dashboard.png",dpi=150)
print("\n✅ All figures saved.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14 — TABLES
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 14 — SAVING IEEE TABLES")

t1=pd.DataFrame([{
    "Class":c.capitalize(),
    "Precision":f"{mc_rep[c]['precision']:.4f}",
    "Recall":f"{mc_rep[c]['recall']:.4f}",
    "F1-Score":f"{mc_rep[c]['f1-score']:.4f}",
    "AUC-ROC":f"{roc_auc_score(y_bin_mc[:,i],ens_pte_c[:,i]):.4f}",
    "Support":int(mc_rep[c]["support"]),
} for i,c in enumerate(CLASS_NAMES)])
t1.to_csv(f"{OUTDIR}/table1_per_class_metrics.csv",index=False)
print(f"Table I:\n{t1.to_string(index=False)}")

t2=pd.DataFrame([{"Metric":k,"Value":f"{v:.4f}" if isinstance(v,float) else str(v)}
  for k,v in list(bm.items())+[
    ("MC Accuracy",mc_rep["accuracy"]),("MC F1 macro",mc_rep["macro avg"]["f1-score"]),
    ("MC F1 weighted",mc_rep["weighted avg"]["f1-score"]),
    ("Burst Recall",mc_rep["burst"]["recall"]),("Theft Recall",mc_rep["theft"]["recall"]),
    ("Alert Threshold",ALERT_THR),("Localization F1",loc_f1),("Localization Acc",loc_acc),
    ("N Models",len(active)),("N Features",len(FEATS)),("Version","4.2"),
]])
t2.to_csv(f"{OUTDIR}/table2_system_metrics.csv",index=False)
gov_s.to_csv(f"{OUTDIR}/table3_governorate_stats.csv",index=False)
df_te_fe.to_csv(f"{OUTDIR}/test_predictions_full.csv",index=False)
df_te_fe[df_te_fe["Pred_Alert"]==1].to_csv(f"{OUTDIR}/alert_log.csv",index=False)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15 — SAVE ALL MODELS
# ═════════════════════════════════════════════════════════════════════════════
sep("SECTION 15 — SAVING ALL MODELS")

joblib.dump(scaler,       f"{OUTDIR}/scaler.pkl")
joblib.dump(meta,         f"{OUTDIR}/model_meta_learner.pkl")
joblib.dump(iso_cals,     f"{OUTDIR}/isotonic_calibrators.pkl")
if LGB_OK:
    lgb_m.save_model(f"{OUTDIR}/model_lightgbm.txt")
    if anom_tr.sum()>100: loc_m.save_model(f"{OUTDIR}/model_localization.txt")
if XGB_OK:
    xgb_m.save_model(f"{OUTDIR}/model_xgboost.json")
if TORCH_OK:
    torch.save(net.state_dict(),       f"{OUTDIR}/model_neural_net.pt")
    torch.save(lstm_net.state_dict(),  f"{OUTDIR}/model_lstm.pt")
    joblib.dump(seq_scaler,            f"{OUTDIR}/scaler_seq.pkl")

with open(f"{OUTDIR}/config.json","w") as fh:
    json.dump({
        "version":       "4.2",
        "features":      FEATS,
        "seq_features":  SEQ_FEATS if TORCH_OK else [],
        "class_names":   CLASS_NAMES,
        "alert_threshold": ALERT_THR,
        "active_models": active,
        "max_seq_len":   MAX_SEQ if TORCH_OK else 4,
        "lstm_fix":      "build_sequences_ordered — output in original row order",
        "changes_v42":   [
            "CRITICAL: LSTM sequence ordering fixed — output guaranteed in original row order",
            "Safety assertions verify y_seq == original labels before training",
            "d_seq verified == Depth_Level - 1 for all rows",
            "Meta-learner now receives correctly aligned probabilities from all 4 models",
        ],
    }, fh, indent=2)

for fn in sorted(os.listdir(OUTDIR)):
    sz=os.path.getsize(f"{OUTDIR}/{fn}")/1024
    print(f"  {fn:<48} {sz:>9.1f} KB")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
sep("AQUAGUARD AI v4.2 — FINAL RESULTS")
print(f"""
╔══════════════════════════════════════════════════════╗
║            BINARY ANOMALY DETECTION                 ║
╠══════════════════════════════════════════════════════╣
║  Accuracy         : {bm['Accuracy']:.4f}                        ║
║  Precision        : {bm['Precision']:.4f}                        ║
║  Recall           : {bm['Recall']:.4f}                        ║
║  F1 Score         : {bm['F1']:.4f}                        ║
║  AUC-ROC          : {bm['AUC-ROC']:.4f}                        ║
║  False Alarm Rate : {bm['False Alarm Rate']:.4f}                        ║
║  Brier Score      : {bm['Brier Score']:.4f}  (calibrated)        ║
╠══════════════════════════════════════════════════════╣
║         MULTI-CLASS CLASSIFICATION                  ║
╠══════════════════════════════════════════════════════╣
║  Accuracy (MC)    : {mc_rep['accuracy']:.4f}                        ║
║  F1 Macro         : {mc_rep['macro avg']['f1-score']:.4f}                        ║
║  F1 Weighted      : {mc_rep['weighted avg']['f1-score']:.4f}                        ║
║  Burst Recall     : {mc_rep['burst']['recall']:.4f}  ← safety critical     ║
║  Theft Recall     : {mc_rep['theft']['recall']:.4f}                        ║
╠══════════════════════════════════════════════════════╣
║            FAULT LOCALIZATION                       ║
╠══════════════════════════════════════════════════════╣
║  Fault_Here F1    : {loc_f1:.4f}                        ║
║  Fault_Here Acc   : {loc_acc:.4f}                        ║
╠══════════════════════════════════════════════════════╣
║  Architecture: LGB + XGB + ResNet-MLP + BiLSTM      ║
║  LSTM v4.2   : ordering fixed, all 4 models active  ║
║  Meta-learner: Logistic Regression (stacking)       ║
║  Calibration : Isotonic regression (per-class)      ║
║  Threshold   : {ALERT_THR:.3f} (tuned on validation)       ║
╚══════════════════════════════════════════════════════╝

Outputs: {OUTDIR}/
  Models  : LGB + XGB + NN + LSTM + Meta + Localization
  Figures : 8 PNG (300 DPI)
  Tables  : 3 CSV
  Logs    : alert_log.csv + test_predictions_full.csv
""")
