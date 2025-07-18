# unsupervised_role_pipeline.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ------------------------------ CONFIG ------------------------------
RAW_LABEL_FILE = "posgrp_labeled_raw.csv"
RAW_PREDICT_FILE = "fbref_big8_raw_2023-2024.csv"
OUT_CSV = "unsupervised_roles.csv"
SAMPLE_OUT = "cluster_examples.csv"
UMAP_POSGRP_PLOT = "viz_posgrp_umap.png"

MIN_MINUTES = 300
KMEANS_K = 3
CORR_THRESHOLD = 0.95
Z_THRESHOLD = 0.75  # Only use z-scores > this for role names
POSGROUPS = ["GK", "CB", "FB", "DM", "AM", "WF", "CF"]
RANDOM_STATE = 42

# ------------------------------ RULE TAGS ---------------------------
def tag_role_from_features(features, zvec, threshold=Z_THRESHOLD):
    tags = {
        "xG": "Finisher", "xAG": "Creator", "PrgP": "Distributor", "PrgC": "Carrier",
        "KP": "Playmaker", "Clr": "Sweeper", "Tkl+Int": "Ball Winner",
        "CrsPA": "Crosser", "Carries_CPA": "Overlapper", "Carries_PrgDist": "Dribbler",
        "Aerial Duels_Won%": "Target", "SCA": "Chance Creator", "Int": "Interceptor",
        "Gls": "Poacher", "Touches_Att Pen": "Advanced", "Blocks_Blocks": "Blocker",
        "Touches_Def Pen": "Deep"
    }
    out = []
    for f in features:
        if abs(zvec[f]) < threshold:
            continue
        if f in tags:
            out.append(tags[f])
    return ", ".join(out[:3]) if out else "Hybrid"

# --------------------- HELPER: drop highly correlated ----------------
def find_highly_correlated_features(df, threshold=0.95, drop=False):
    df_corr = df.corr().abs()
    seen = set()
    correlated_groups = []

    for col in df_corr.columns:
        if col in seen:
            continue
        high_corr = df_corr[col][df_corr[col] > threshold].index.tolist()
        high_corr = [x for x in high_corr if x != col]
        if high_corr:
            group = [col] + high_corr
            seen.update(group)
            correlated_groups.append(group)

    if drop:
        to_drop = [f for group in correlated_groups for f in group[1:]]
        df = df.drop(columns=to_drop)

    return df, correlated_groups

# -------------------- 1. Load labeled + raw -------------------------
df_labeled = pd.read_csv(RAW_LABEL_FILE)
df_pred = pd.read_csv(RAW_PREDICT_FILE)
df_pred = df_pred[pd.to_numeric(df_pred["Min"], errors="coerce") >= MIN_MINUTES]

DROP_COLS = {
    "Player","Nation","Squad","League","Pos","Role","PosGrp",
    "Age","Min","MP","Starts","Gls","Ast","G+A","G-PK","PK","PKatt",
    "CrdY","CrdR","Err"
}
feature_cols = [c for c in df_labeled.columns if c not in DROP_COLS]
X_df = df_labeled[feature_cols].apply(pd.to_numeric, errors="coerce")
X_df = pd.DataFrame(SimpleImputer().fit_transform(X_df), columns=feature_cols)

X_df, corr_groups = find_highly_correlated_features(X_df, threshold=CORR_THRESHOLD, drop=True)
feature_cols = X_df.columns.tolist()
X_numeric = X_df.to_numpy()

# --------------------- 2. Train PosGrp model ------------------------
y_pos = df_labeled["PosGrp"].values
le = LabelEncoder()
y_enc = le.fit_transform(y_pos)

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y_enc, stratify=y_enc, random_state=42
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=400,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# Accuracy
acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"\n✅ PosGrp Model Accuracy: {acc:.3%}")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, digits=3))

joblib.dump(model, "posgrp_xgb_model.pkl")
joblib.dump(le, "posgrp_label_encoder.pkl")

# -------------------- 3. Predict PosGrp for full dataset --------------------
X_pred = df_pred[feature_cols].apply(pd.to_numeric, errors="coerce")
X_pred = pd.DataFrame(SimpleImputer().fit_transform(X_pred), columns=feature_cols)

df_pred["PosGrp"] = le.inverse_transform(model.predict(X_pred))

# -------------------- 4. UMAP for PosGrp Visualization --------------------
if HAS_UMAP:
    scaler_pos = StandardScaler().fit(X_pred)
    X_pos_std = scaler_pos.transform(X_pred)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
    umap_pos = reducer.fit_transform(X_pos_std)
    df_pred["UMAP_X"] = umap_pos[:, 0]
    df_pred["UMAP_Y"] = umap_pos[:, 1]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pred, x="UMAP_X", y="UMAP_Y", hue="PosGrp", palette="tab10", s=40)
    plt.title("UMAP: Position Groups")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(UMAP_POSGRP_PLOT)
    plt.close()

# -------------------- 5. Per-PosGrp Clustering + Role Naming --------------------
out_dfs = []
samples = []

for pg in POSGROUPS:
    sub = df_pred[df_pred["PosGrp"] == pg].copy().reset_index(drop=True)
    if sub.empty: continue

    X_pg = sub[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_pg = pd.DataFrame(SimpleImputer().fit_transform(X_pg), columns=feature_cols).reset_index(drop=True)
    X_pg_std = StandardScaler().fit_transform(X_pg)

    km = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE)
    cluster_labels = km.fit_predict(X_pg_std)
    sub["RoleCluster"] = cluster_labels

    centroids = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    centroid_z = centroids.apply(lambda x: (x - x.mean()) / (x.std() + 1e-6), axis=0)

    role_names = []
    top_feats = []

    for i in range(KMEANS_K):
        row = centroid_z.iloc[i]
        top = row.abs().sort_values(ascending=False).head(5).index.tolist()
        desc = ", ".join([f"{t}↑" if row[t] > 0 else f"{t}↓" for t in top])
        tag = tag_role_from_features(top, row)
        role_names.append(f"{pg}-{tag}")
        top_feats.append(desc)

    role_map = dict(zip(range(KMEANS_K), role_names))
    feat_map = dict(zip(range(KMEANS_K), top_feats))

    sub["SuggestedRole"] = sub["RoleCluster"].map(role_map)
    sub["TopFeatures"] = sub["RoleCluster"].map(feat_map)

    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
        emb = reducer.fit_transform(X_pg_std)
        sub["UMAP_X"] = emb[:,0]
        sub["UMAP_Y"] = emb[:,1]

        plt.figure(figsize=(6,5))
        sns.scatterplot(data=sub, x="UMAP_X", y="UMAP_Y",
                        hue="SuggestedRole", palette="tab10", s=40, alpha=0.8)
        plt.title(f"{pg} Role Clusters")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"viz_{pg}_roles.png")
        plt.close()

    # Sample top 5 players per cluster
    for i in range(KMEANS_K):
        role_df = sub[sub["RoleCluster"] == i]
        # Get player vectors and centroid
        role_df = role_df.copy().reset_index(drop=True)
        X_cluster = X_pg_std[role_df.index]
        centroid = km.cluster_centers_[i]

        # Compute distances to centroid
        dists = np.linalg.norm(X_cluster - centroid, axis=1)
        closest_idx = np.argsort(dists)[:5]

        # Select closest players
        top_players = role_df.iloc[closest_idx]

        for _, row in top_players.iterrows():
            samples.append({
                "PosGrp": pg,
                "SuggestedRole": role_map[i],
                "TopFeatures": feat_map[i],
                "Player": row["Player"],
                "Squad": row["Squad"],
                "League": row["League"]
            })

    out_dfs.append(sub)

# --------------------- 6. Export -----------------------------
final_df = pd.concat(out_dfs, ignore_index=True)
cols_out = ["Player", "Squad", "League", "Pos", "PosGrp", "RoleCluster",
            "SuggestedRole", "TopFeatures"]
if HAS_UMAP:
    cols_out += ["UMAP_X", "UMAP_Y"]

final_df[cols_out].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(samples).to_csv(SAMPLE_OUT, index=False, encoding="utf-8-sig")

print(f"✓ Saved {len(final_df)} player-role predictions → {OUT_CSV}")
print(f"✓ Saved top cluster examples → {SAMPLE_OUT}")
