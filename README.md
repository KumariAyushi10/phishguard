# phishguard

A machine learning web application that predicts whether a given URL is potentially phishing or legitimate by analyzing its structural tokens and cross-referencing live security threats via the VirusTotal API.

## Features

* **VirusTotal API Integration:** Connects seamlessly to the VirusTotal API to cross-reference URLs against real-time global threat databases.
* **Risk Classification:** Predicts whether a URL is a phishing threat (0) or legitimate link (1) instantly.
* **Frictionless Analysis:** Extracts high-signal URL characteristics directly from text input without requiring a complete website network crawl.
* **Predictive ML Engine:** Utilizes a trained data science classifier to evaluate security risks dynamically based on historical dataset trends.

## Project Structure

```text
├── static/             
├── templates/          
├── .gitignore          
├── app.py              
├── requirements.txt    
└── README.md           
```

## How to Run It

1. Clone or download the repository and navigate to the folder.
2. Set up a virtual environment and install dependencies via `pip install -r requirements.txt`.
3. Run `python app.py` and open `http://127.0.0.1:5000` in your web browser.


## AI / ML Engine

PhishGuard now combines VirusTotal with its own machine-learning model. Everything above (dashboard design, VirusTotal scan, history) works exactly as before; the AI layer is added on top.

### What was added

* **URL classifier** – Gradient Boosting model (selected automatically over Logistic Regression and Random Forest) trained on ~355k labelled URLs.
* **46 engineered features** – URL/host/path length, subdomain count, IP-as-host, suspicious TLDs, phishing keywords, brand impersonation (e.g. `paypal` in a domain that isn't PayPal), Shannon entropy, free-hosting detection, domain popularity rank and more (`ml/features.py`).
* **Hybrid verdict** – AI probability is blended with VirusTotal engine counts. If VirusTotal fails or has no data, the AI model still gives a verdict.
* **Explainable AI** – each result shows which features pushed the score towards phishing or towards legitimate.
* **Model dashboard** – accuracy, precision, recall, F1, confusion matrix, ROC curve, algorithm comparison and global feature importance ("AI Engine" section).
* **Feedback loop** – "This URL is phishing / legitimate" buttons store labels in `ml/data/feedback.csv`; they are merged in at the next training run.
* **JSON API** – `POST /api/predict` (`{"url": "https://..."}`), `GET /api/model-info`, `POST /api/feedback`.

### Project structure (new files)

```text
├── ml/
│   ├── features.py       # URL feature extraction
│   ├── train.py          # data download, training, evaluation
│   ├── predictor.py      # loading, prediction, explanation, hybrid verdict
│   └── data/top_domains.txt   # popularity list used by the domain-rank features
└── models/
    ├── phishguard_model.joblib  # trained model
    └── metrics.json             # metrics shown in the dashboard
```

### Re-training

```bash
python -m ml.train --download          # download public datasets, train, evaluate, save
python -m ml.train                     # re-train (also picks up feedback.csv)
python -m ml.train --csv my_urls.csv  # your own data: columns url,label (phishing / legitimate)
```

Datasets: [Phishing.Database](https://github.com/mitchellkrogza/Phishing.Database) (phishing URLs), [faizann24's URL dataset](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs) (good/bad URLs) and a public top-domains list. Train/test are split **by registered domain**, so reported scores reflect performance on websites the model has never seen.

### Results (held-out test set, unseen domains)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 93.2% | 93.5% | 91.8% | 92.7% | 0.984 |

### Limitations

* The model only reads the URL text, so it can miss phishing on compromised legitimate sites and can flag unusual legitimate URLs. Treat it as one signal, not a guarantee; that is why it is paired with VirusTotal.
* Training data is about 44% phishing, far more than real traffic, so the probability is best read as a relative risk score.
* The saved model is a scikit-learn artifact. If you use a very different scikit-learn version and loading fails, the app shows a notice and you can simply re-train.
* In this AI module labels are written explicitly as `phishing` / `legitimate` (API returns `is_phishing`) instead of the 0/1 codes.
