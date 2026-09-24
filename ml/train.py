"""
Train the PhishGuard URL classifier.

Usage
-----
    python -m ml.train --download            # fetch public datasets, train, save
    python -m ml.train                       # re-train from data already in ml/data/
    python -m ml.train --csv my_urls.csv     # use your own CSV (columns: url,label)

Datasets (downloaded from public GitHub repositories):
  * Phishing.Database  - recent, verified phishing URLs   (label: phishing)
  * faizann24 data.csv - ~420k labelled URLs, good and bad (good -> legitimate,
                         bad -> phishing/malicious)

Extra labelled rows collected through the app's feedback buttons
(ml/data/feedback.csv) are automatically merged into the training set.

Outputs
-------
    models/phishguard_model.joblib   trained model + metadata
    models/metrics.json              evaluation metrics shown in the dashboard
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import (  # noqa: E402
    FEATURE_LABELS, FEATURE_NAMES, extract_features, features_to_vector,
    normalize_url, registered_domain,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "ml", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishguard_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.csv")

SOURCES = {
    "phishing_db.txt": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "top_domains_raw.txt": "https://raw.githubusercontent.com/zer0h/top-1000000-domains/master/top-1000000-domains",
    "faizann_data.csv": "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv",
}
SEED = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_sources():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[data] {name} already present, skipping download")
            continue
        print(f"[data] downloading {name} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    build_top_domains()


def build_top_domains(limit=200_000):
    """Write the popularity list used by the 'domain rank' features."""
    raw = os.path.join(RAW_DIR, "top_domains_raw.txt")
    if not os.path.exists(raw):
        return
    with open(raw, encoding="utf-8", errors="ignore") as fh:
        top = [ln.strip().lower() for ln in fh if ln.strip()][:limit]
    with open(os.path.join(DATA_DIR, "top_domains.txt"), "w", encoding="utf-8") as out:
        out.write("\n".join(top) + "\n")
    print(f"[data] wrote top_domains.txt ({len(top):,} domains)")


def _sample(df, n):
    return df if len(df) <= n else df.sample(n=n, random_state=SEED)


def load_public_data(phish_db_n, faizann_bad_n, good_n):
    frames = []
    pdb = os.path.join(RAW_DIR, "phishing_db.txt")
    fz = os.path.join(RAW_DIR, "faizann_data.csv")
    if not (os.path.exists(pdb) and os.path.exists(fz)):
        raise FileNotFoundError(
            "Public datasets not found in ml/data/raw. Run: python -m ml.train --download"
        )

    with open(pdb, encoding="utf-8", errors="ignore") as fh:
        urls = [ln.strip() for ln in fh if ln.lower().startswith(("http://", "https://"))]
    d = pd.DataFrame({"url": urls, "label": "phishing", "source": "phishing_db"})
    frames.append(_sample(d.drop_duplicates("url"), phish_db_n))

    f = pd.read_csv(fz, on_bad_lines="skip").dropna()
    f["label"] = f["label"].map({"bad": "phishing", "good": "legitimate"})
    f["source"] = "faizann"
    f = f.dropna(subset=["label"])
    frames.append(_sample(f[f.label == "phishing"], faizann_bad_n))
    frames.append(_sample(f[f.label == "legitimate"], good_n))
    return pd.concat(frames, ignore_index=True)


def load_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return pd.DataFrame(columns=["url", "label", "source"])
    fb = pd.read_csv(FEEDBACK_PATH)
    fb = fb[fb["label"].isin(["phishing", "legitimate"])][["url", "label"]].copy()
    fb["source"] = "feedback"
    return fb


def build_dataset(args):
    if args.csv:
        df = pd.read_csv(args.csv)[["url", "label"]].dropna()
        df["label"] = df["label"].astype(str).str.lower().replace(
            {"1": "phishing", "bad": "phishing", "malicious": "phishing",
             "0": "legitimate", "good": "legitimate", "benign": "legitimate"})
        df["source"] = "custom_csv"
    else:
        df = load_public_data(args.phish_db, args.faizann_bad, args.good)
    df = pd.concat([df, load_feedback()], ignore_index=True)

    df["norm"] = df["url"].map(normalize_url)
    df = df[df["norm"].str.len().between(4, 2000)]
    df = df.drop_duplicates("norm").reset_index(drop=True)
    df["y"] = (df["label"] == "phishing").astype(int)
    df["group"] = df["url"].map(registered_domain)
    return df


def featurize(urls):
    rows = [features_to_vector(extract_features(u)) for u in urls]
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Models / evaluation
# --------------------------------------------------------------------------- #
def candidate_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=22, min_samples_leaf=3,
            n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=48,
            early_stopping=True, random_state=SEED),
    }


def score(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, proba, pred


def group_split(df, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, df["y"], groups=df["group"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download public datasets first")
    ap.add_argument("--csv", help="custom CSV with columns url,label")
    ap.add_argument("--phish-db", type=int, default=120_000, help="rows from Phishing.Database")
    ap.add_argument("--faizann-bad", type=int, default=40_000, help="malicious rows from faizann data")
    ap.add_argument("--good", type=int, default=200_000, help="legitimate rows")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    if args.download and not args.csv:
        download_sources()

    df = build_dataset(args)
    print(f"[data] {len(df):,} unique URLs  |  phishing={df.y.sum():,}  legitimate={(1 - df.y).sum():,}")

    print("[features] extracting ...")
    X = featurize(df["url"])
    y = df["y"].values

    # Split by registered domain so the same site never appears in both train and test
    trainval_idx, test_idx = group_split(df, 0.15, SEED)
    sub = df.iloc[trainval_idx].reset_index(drop=True)
    tr_rel, va_rel = group_split(sub, 0.15 / 0.85, SEED)
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]
    print(f"[split] train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,} (grouped by domain)")

    # ---- compare candidates on validation set --------------------------------
    comparison, fitted = [], {}
    for name, model in candidate_models().items():
        s = time.time()
        model.fit(X[train_idx], y[train_idx])
        val_m, _, _ = score(model, X[val_idx], y[val_idx])
        test_m, _, _ = score(model, X[test_idx], y[test_idx])
        fitted[name] = model
        comparison.append({"name": name, "val": val_m, "test": test_m,
                           "train_seconds": round(time.time() - s, 1)})
        print(f"[model] {name:<20} val F1={val_m['f1']:.4f} AUC={val_m['roc_auc']:.4f} "
              f"| test F1={test_m['f1']:.4f}  ({time.time() - s:.0f}s)")

    best_name = max(comparison, key=lambda c: (c["val"]["roc_auc"], c["val"]["f1"]))["name"]
    print(f"[model] selected: {best_name}")

    # ---- refit best on train+val, final evaluation on untouched test set -----
    final_idx = np.concatenate([train_idx, val_idx])
    best = candidate_models()[best_name]
    best.fit(X[final_idx], y[final_idx])
    final_m, proba, pred = score(best, X[test_idx], y[test_idx])
    tn, fp, fn, tp = confusion_matrix(y[test_idx], pred).ravel()

    fpr, tpr, _ = roc_curve(y[test_idx], proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
    roc_points = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep]

    # per-source recall on the test set (how well do we do on each data source?)
    per_source = {}
    test_df = df.iloc[test_idx].assign(pred=pred)
    for src, g in test_df.groupby("source"):
        pos = g[g.y == 1]
        per_source[src] = {
            "rows": int(len(g)),
            "accuracy": round(float((g.pred == g.y).mean()), 4),
            "phishing_recall": round(float((pos.pred == 1).mean()), 4) if len(pos) else None,
        }

    # global feature importance (permutation, on a test subsample)
    rng = np.random.RandomState(SEED)
    sample = rng.choice(len(test_idx), size=min(15_000, len(test_idx)), replace=False)
    perm = permutation_importance(
        best, X[test_idx][sample], y[test_idx][sample],
        scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1)
    order = np.argsort(perm.importances_mean)[::-1][:12]
    importances = [
        {"feature": FEATURE_NAMES[i], "label": FEATURE_LABELS[FEATURE_NAMES[i]],
         "importance": round(float(perm.importances_mean[i]), 5)}
        for i in order
    ]

    # Baseline vectors = per-class medians (used for local explanations)
    baseline = np.median(X[final_idx][y[final_idx] == 0], axis=0)
    baseline_phish = np.median(X[final_idx][y[final_idx] == 1], axis=0)

    metrics = {
        "model_name": best_name,
        "trained_at": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "dataset": {
            "total": int(len(df)),
            "phishing": int(df.y.sum()),
            "legitimate": int((1 - df.y).sum()),
            "train": int(len(final_idx)),
            "test": int(len(test_idx)),
            "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
            "split": "grouped by registered domain",
        },
        "n_features": len(FEATURE_NAMES),
        "test": {k: round(float(v), 4) for k, v in final_m.items()},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": roc_points,
        "importances": importances,
        "per_source": per_source,
        "comparison": [
            {"name": c["name"],
             "accuracy": round(c["test"]["accuracy"], 4),
             "f1": round(c["test"]["f1"], 4),
             "roc_auc": round(c["test"]["roc_auc"], 4),
             "selected": c["name"] == best_name}
            for c in comparison
        ],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": best,
        "feature_names": FEATURE_NAMES,
        "baseline": baseline.tolist(),
        "baseline_phish": baseline_phish.tolist(),
        "model_name": best_name,
        "trained_at": metrics["trained_at"],
    }, MODEL_PATH, compress=3)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== Final test-set performance (unseen domains) ===")
    for k, v in metrics["test"].items():
        print(f"  {k:<10} {v:.4f}")
    print(f"  confusion  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"[done] saved {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
