
from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os
import time
import base64
from dotenv import load_dotenv
from datetime import datetime

# --- AI/ML engine (added) ---
from ml.predictor import get_model, combine, save_feedback

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

VT_BASE_URL = "https://www.virustotal.com/api/v3"

# Cached VirusTotal reports older than this are re-scanned (24 hours)
REPORT_MAX_AGE_SECONDS = 24 * 60 * 60

history = []

phish_model = get_model()

# Running totals for the overview cards. They are kept separately because
# `history` only holds the latest 10 scans, which made the cards stop updating.
session_totals = {"total": 0, "threat": 0, "clean": 0}


def update_session_totals(result, combined):
    """Count a finished scan (VirusTotal and AI verdicts both considered)."""
    vt_flagged = result["label"] in ["Potentially Malicious", "Suspicious"]
    ai_flagged = bool(combined) and combined["label"] in [
        "Likely Phishing", "Suspicious"
    ]

    if result["label"] in ["Scan Error", "No Data Available"] and not combined:
        return

    session_totals["total"] += 1

    if vt_flagged or ai_flagged:
        session_totals["threat"] += 1
    elif result["label"] == "No Malicious Detection" or combined:
        session_totals["clean"] += 1


def get_url_id(url):
    encoded_url = base64.urlsafe_b64encode(
        url.encode()
    ).decode().strip("=")

    return encoded_url


def get_existing_report(url):
    url_id = get_url_id(url)

    headers = {
        "x-apikey": API_KEY
    }

    response = requests.get(
        f"{VT_BASE_URL}/urls/{url_id}",
        headers=headers,
        timeout=20
    )

    if response.status_code == 200:
        data = response.json()
        attributes = data["data"]["attributes"]
        analysed_at = attributes.get("last_analysis_date")
        age = (time.time() - analysed_at) if analysed_at else None
        return attributes.get("last_analysis_stats", {}), age

    return None, None


def scan_url(url):
    if not API_KEY:
        return {
            "error": "VirusTotal API key is missing."
        }

    headers = {
        "x-apikey": API_KEY
    }

    # First, check whether VirusTotal already has a *recent* report.
    # Old reports can be outdated (a URL may have been flagged since), so
    # reports older than REPORT_MAX_AGE_SECONDS trigger a fresh scan.
    stale_stats = None

    try:
        existing_stats, age = get_existing_report(url)

        if existing_stats:
            if age is not None and age <= REPORT_MAX_AGE_SECONDS:
                return existing_stats

            stale_stats = existing_stats

    except requests.exceptions.RequestException:
        pass

    result = submit_for_scan(url, headers)

    # If the re-scan failed, fall back to the older report instead of an error
    if "error" in result and stale_stats:
        return stale_stats

    return result


def submit_for_scan(url, headers):
    # Submit URL for a new scan
    try:
        response = requests.post(
            f"{VT_BASE_URL}/urls",
            headers=headers,
            data={"url": url},
            timeout=20
        )

        if response.status_code == 401:
            return {
                "error": "Invalid VirusTotal API key."
            }

        if response.status_code == 429:
            return {
                "error": "VirusTotal API rate limit reached. Try again later."
            }

        if response.status_code != 200:
            return {
                "error": f"VirusTotal error: {response.status_code}"
            }

        result = response.json()

        analysis_id = result["data"]["id"]

        # Wait for the scan to finish
        for _ in range(10):
            time.sleep(3)

            analysis_response = requests.get(
                f"{VT_BASE_URL}/analyses/{analysis_id}",
                headers=headers,
                timeout=20
            )

            if analysis_response.status_code != 200:
                continue

            analysis_data = analysis_response.json()
            attributes = analysis_data["data"]["attributes"]

            status = attributes.get("status")

            if status == "completed":
                return attributes.get("stats", {})

        return {
            "error": "The scan is taking too long. Please try again."
        }

    except requests.exceptions.Timeout:
        return {
            "error": "VirusTotal request timed out."
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": f"Connection error: {str(error)}"
        }

    except Exception as error:
        return {
            "error": f"Unexpected error: {str(error)}"
        }


def create_result(stats):
    if "error" in stats:
        return {
            "label": "Scan Error",
            "risk_level": "Unknown",
            "style": "danger",
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "total": 0,
            "detection_rate": "N/A",
            "method": "VirusTotal API",
            "message": stats["error"]
        }

    malicious = stats.get("malicious", 0)
    suspicious = stats.get("suspicious", 0)
    harmless = stats.get("harmless", 0)
    undetected = stats.get("undetected", 0)

    total = malicious + suspicious + harmless + undetected

    if total > 0:
        detection_rate = round(
            ((malicious + suspicious) / total) * 100,
            2
        )
    else:
        detection_rate = 0

    if malicious > 0:
        label = "Potentially Malicious"
        risk_level = "High Risk"
        style = "danger"
        message = (
            "One or more security engines detected "
            "potentially malicious behavior."
        )

    elif suspicious > 0:
        label = "Suspicious"
        risk_level = "Medium Risk"
        style = "warning"
        message = (
            "Some security engines reported suspicious "
            "characteristics."
        )

    elif total == 0:
        label = "No Data Available"
        risk_level = "Unknown"
        style = "warning"
        message = (
            "VirusTotal did not return sufficient analysis data."
        )

    else:
        label = "No Malicious Detection"
        risk_level = "Low Risk"
        style = "safe"
        message = (
            "No malicious or suspicious detections were "
            "reported by the available security engines."
        )

    return {
        "label": label,
        "risk_level": risk_level,
        "style": style,
        "malicious": malicious,
        "suspicious": suspicious,
        "harmless": harmless,
        "undetected": undetected,
        "total": total,
        "detection_rate": detection_rate,
        "method": "VirusTotal API",
        "message": message
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    scanned_url = ""
    ml_result = None
    combined = None

    if request.method == "POST":
        scanned_url = request.form.get("url", "").strip()

        if not scanned_url:
            result = {
                "label": "Invalid Input",
                "risk_level": "Unknown",
                "style": "warning",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "total": 0,
                "detection_rate": "N/A",
                "method": "VirusTotal API",
                "message": "Please enter a URL."
            }

        elif not (
            scanned_url.startswith("http://")
            or scanned_url.startswith("https://")
        ):
            result = {
                "label": "Invalid URL",
                "risk_level": "Unknown",
                "style": "warning",
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "total": 0,
                "detection_rate": "N/A",
                "method": "VirusTotal API",
                "message": (
                    "Please enter a complete URL beginning "
                    "with http:// or https://."
                )
            }

        else:
            stats = scan_url(scanned_url)
            result = create_result(stats)

            # AI/ML analysis runs alongside VirusTotal (and still works if VT fails)
            ml_result = phish_model.predict(scanned_url)
            combined = combine(ml_result, result)
            update_session_totals(result, combined)

            history.insert(0, {
                "url": scanned_url,
                "label": result["label"],
                "risk_level": result["risk_level"],
                "detection_rate": result["detection_rate"],
                "time": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "ai_score": combined["score"] if combined else None,
                "ai_verdict": combined["label"] if combined else None
            })

            if len(history) > 10:
                history.pop()

    statistics = {
        "total_scans": session_totals["total"],
        "phishing_scans": session_totals["threat"],
        "legitimate_scans": session_totals["clean"]
    }

    return render_template(
        "index.html",
        result=result,
        scanned_url=scanned_url,
        history=history,
        statistics=statistics,
        ml=ml_result,
        combined=combined,
        model_info=phish_model.metrics if phish_model.ready else None,
        model_error=None if phish_model.ready else phish_model.error
    )


@app.route("/clear-history")
def clear_history():
    history.clear()
    session_totals.update(total=0, threat=0, clean=0)
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# AI/ML JSON API (added)
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    """ML-only prediction (no VirusTotal call): fast and needs no API key."""
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or request.values.get("url") or "").strip()

    if not url.startswith(("http://", "https://")):
        return jsonify({
            "error": "Provide a complete URL beginning with http:// or https://."
        }), 400

    result = phish_model.predict(url)

    if not result.get("available"):
        return jsonify({"error": result["message"]}), 503

    return jsonify(result)


@app.route("/api/model-info")
def api_model_info():
    if not phish_model.ready:
        return jsonify({"error": phish_model.error}), 503

    return jsonify(phish_model.metrics)


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Store a human label so the model can be re-trained with it."""
    payload = request.get_json(silent=True) or {}

    try:
        save_feedback(
            payload.get("url"),
            payload.get("label"),
            payload.get("probability")
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"status": "saved"})


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )