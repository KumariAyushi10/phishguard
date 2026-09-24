"""
Runtime side of the PhishGuard AI engine: load the trained model, score a URL,
explain the prediction and fuse it with the VirusTotal verdict.
"""
import csv
import json
import os
import threading
from datetime import datetime

import joblib
import numpy as np

from ml.features import (
    FEATURE_LABELS, FEATURE_NAMES, extract_features, features_to_vector,
    format_value, normalize_url,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "phishguard_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
FEEDBACK_PATH = os.path.join(ROOT, "ml", "data", "feedback.csv")

HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.30
_feedback_lock = threading.Lock()


class PhishingModel:
    """Thin wrapper around the persisted scikit-learn model."""

    def __init__(self):
        self.bundle = None
        self.metrics = None
        self.error = None
        self._load()

    def _load(self):
        try:
            self.bundle = joblib.load(MODEL_PATH)
            if self.bundle["feature_names"] != FEATURE_NAMES:
                raise ValueError("model was trained with a different feature set")
            self.baseline = np.asarray(self.bundle["baseline"], dtype=np.float32)
            self.baseline_phish = np.asarray(self.bundle["baseline_phish"], dtype=np.float32)
        except FileNotFoundError:
            self.error = "No trained model found. Run: python -m ml.train --download"
            self.bundle = None
        except Exception as exc:                        # version mismatch, corrupt file...
            self.error = f"Model could not be loaded ({exc}). Re-train with: python -m ml.train"
            self.bundle = None
        try:
            with open(METRICS_PATH) as fh:
                self.metrics = json.load(fh)
        except (FileNotFoundError, ValueError):
            self.metrics = None

    @property
    def ready(self):
        return self.bundle is not None

    # ------------------------------------------------------------------ #
    def predict(self, url, explain=True, top_k=5):
        """Score one URL. Returns a JSON-serialisable dict."""
        if not self.ready:
            return {"available": False, "message": self.error}

        feats = extract_features(url)
        x = np.asarray([features_to_vector(feats)], dtype=np.float32)
        model = self.bundle["model"]
        p = float(model.predict_proba(x)[0, 1])

        if p >= HIGH_THRESHOLD:
            risk, style = "High Risk", "danger"
        elif p >= MEDIUM_THRESHOLD:
            risk, style = "Medium Risk", "warning"
        else:
            risk, style = "Low Risk", "safe"

        is_phishing = p >= 0.5
        out = {
            "available": True,
            "url": url,
            "is_phishing": is_phishing,
            "label": "Likely Phishing" if is_phishing else "Likely Legitimate",
            "phishing_probability": round(p * 100, 1),
            "confidence": round(max(p, 1 - p) * 100, 1),
            "risk_level": risk,
            "style": style,
            "model_name": self.bundle["model_name"],
            "factors": {"risk": [], "reassuring": []},
        }
        if explain:
            out["factors"] = self._explain(x[0], p, feats, top_k)
        return out

    def _explain(self, x, p, feats, top_k):
        """
        Local explanation by occlusion in log-odds space.

        * Risk signals: reset a feature to the value typical of legitimate URLs;
          if the phishing score drops, that feature was pushing towards phishing.
        * Reassuring signals: reset a feature to the value typical of phishing
          URLs; if the score rises, that feature was pushing towards legitimate.

        Working in log-odds keeps the explanation informative even when the
        probability is saturated near 0% or 100%.
        """
        model = self.bundle["model"]
        n = len(x)
        # features that describe the same thing are occluded together
        coupled = {"domain_rank_log": "in_top_domains"}
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

        def variants_for(baseline):
            v = np.tile(x, (n, 1))
            for i in range(n):
                v[i, i] = baseline[i]
                partner = coupled.get(FEATURE_NAMES[i])
                if partner:
                    v[i, idx[partner]] = baseline[idx[partner]]
            return v

        base_logit = _logit(p)
        risk = base_logit - _logit(model.predict_proba(variants_for(self.baseline))[:, 1])
        safe = _logit(model.predict_proba(variants_for(self.baseline_phish))[:, 1]) - base_logit

        def pick(deltas, direction, limit):
            out = []
            for i in np.argsort(-deltas):
                name = FEATURE_NAMES[i]
                if name in coupled.values() or deltas[i] < 0.15 or len(out) >= limit:
                    continue
                d = float(deltas[i])
                out.append({
                    "feature": name,
                    "label": FEATURE_LABELS[name] if name != "domain_rank_log"
                    else "Domain popularity",
                    "value": format_value(name, feats[name]),
                    "strength": "Strong" if d >= 2 else "Moderate" if d >= 0.7 else "Slight",
                    "width": int(min(100, round(d / 4 * 100))),
                    "direction": direction,
                })
            return out

        return {
            "risk": pick(risk, "phishing", top_k),
            "reassuring": pick(safe, "legitimate", max(top_k - 2, 2)),
        }


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------- #
# Fusion of ML + VirusTotal
# ---------------------------------------------------------------------- #
def combine(ml, vt_result):
    """
    Blend the ML probability with the VirusTotal engine counts into one verdict.

    vt_result is the dict produced by app.create_result(); it is unusable when
    VirusTotal returned an error or no engine data.
    """
    if not ml or not ml.get("available"):
        return None

    p = ml["phishing_probability"] / 100
    vt_ok = bool(vt_result) and vt_result.get("total", 0) > 0 \
        and vt_result.get("label") not in ("Scan Error", "Invalid URL", "Invalid Input")

    if vt_ok:
        mal, sus = vt_result["malicious"], vt_result["suspicious"]
        vt_score = min(1.0, (mal + 0.5 * sus) / 3)
        score = 0.6 * vt_score + 0.4 * p
        if mal >= 3:
            score = max(score, 0.85)
        # VirusTotal often has not seen brand-new phishing yet: when the AI is
        # confident, engines reporting "clean" must not turn it into "safe".
        if p >= 0.95:
            score = max(score, 0.55)
        elif p >= 0.75:
            score = max(score, 0.40)
        elif p >= 0.5:
            score = max(score, 0.30)
        sources = "VirusTotal + AI model"
        if vt_score >= 0.3 and p >= 0.5:
            note = "Both VirusTotal and the AI model flag this URL. Treat it as dangerous."
        elif vt_score >= 0.3:
            note = ("VirusTotal engines flag this URL although its structure looks "
                    "ordinary to the AI model (e.g. a compromised legitimate site).")
        elif p >= 0.5:
            note = ("The AI model finds phishing-like patterns, but no engine has flagged "
                    "the URL yet. Newly created phishing pages often look like this.")
        else:
            note = "Neither VirusTotal nor the AI model found signs of phishing."
    else:
        score = p
        sources = "AI model only"
        note = ("VirusTotal data was unavailable, so this verdict relies on the "
                "AI model alone.")

    if score >= HIGH_THRESHOLD:
        risk, style, label = "High Risk", "danger", "Likely Phishing"
    elif score >= MEDIUM_THRESHOLD:
        risk, style, label = "Medium Risk", "warning", "Suspicious"
    else:
        risk, style, label = "Low Risk", "safe", "Likely Safe"

    return {
        "score": round(score * 100, 1),
        "risk_level": risk,
        "style": style,
        "label": label,
        "sources": sources,
        "note": note,
    }


# ---------------------------------------------------------------------- #
# Human feedback (used for re-training)
# ---------------------------------------------------------------------- #
def save_feedback(url, label, model_probability=None):
    """Append a user-provided label to ml/data/feedback.csv."""
    if label not in ("phishing", "legitimate"):
        raise ValueError("label must be 'phishing' or 'legitimate'")
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")) or len(url) > 2000:
        raise ValueError("invalid url")
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with _feedback_lock:
        new_file = not os.path.exists(FEEDBACK_PATH)
        with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["url", "label", "model_probability", "time"])
            w.writerow([url, label, model_probability,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


_model = None


def get_model():
    global _model
    if _model is None:
        _model = PhishingModel()
    return _model
