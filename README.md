# phishguard

A machine learning project that detects phishing URLs using a Gradient Boosting classifier trained on about 355k labelled URLs, with explainable predictions and VirusTotal threat intelligence as a second opinion, built with Python and Flask.

## Features

* **VirusTotal API Integration:** Cross-references URLs against real-time global threat databases and shows engine-by-engine detection counts.
* **ML Phishing Classifier:** A Gradient Boosting model, selected over Logistic Regression and Random Forest, trained on about 355k labelled URLs. It scores around 93% accuracy on domains it has never seen.
* **Feature Engineering:** Extracts 46 features from the URL text alone, such as length, subdomains, IP hosts, suspicious TLDs, phishing keywords, brand impersonation, entropy and domain popularity. No page crawl is needed.
* **Explainable AI:** Shows which signals pushed each URL towards phishing and which pushed it towards legitimate.
* **Hybrid Verdict:** Blends the AI probability with the VirusTotal results, and still returns an AI-only verdict if VirusTotal is unavailable.
* **Feedback Loop and API:** Users can label results as phishing or legitimate for the next training run, and a JSON API (`/api/predict`) is available.

## Project Structure

```
├── ml/
│   ├── features.py     
│   ├── train.py        
│   ├── predictor.py    
│   └── data/           
├── models/             
├── templates/          
├── static/             
├── app.py              
├── requirements.txt    
└── README.md           
```

## How to Run It

1. Clone or download the repository and navigate to the folder.
2. Set up a virtual environment and install dependencies via `pip install -r requirements.txt`.
3. Create a `.env` file containing `VIRUSTOTAL_API_KEY=your_key_here`.
4. Train the model with `python -m ml.train --download`.
5. Run `python app.py` and open `http://127.0.0.1:5000` in your browser.
