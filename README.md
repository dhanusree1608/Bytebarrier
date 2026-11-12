# 🛡️ ByteBarrier – Multilingual Scam Message Checker
**Live demo:** [https://bytebarrier-scamcheck.streamlit.app](https://bytebarrier-scamcheck.streamlit.app)
ByteBarrier helps protect senior citizens from phone and message fraud. It checks a typed or pasted message in **English, Tamil and Tanglish**, shows a **0–100 risk score**, explains *why* it was flagged, gives advice in Tamil, English or Hindi, and offers a one-tap **call / WhatsApp link to a trusted person**.

## How it works
- **Machine learning:** TF-IDF character n-grams + Logistic Regression, trained on 270 hand-written fictional sentences (135 Scam, 135 Normal; 90 per language).
- **Key-points rules layer:** flags OTP/PIN requests, bank/KYC details, urgency, threats, money demands, prize bait, remote-app requests and links.
- **Final score:** the higher of the rules score and the model score, so obvious scams are caught even in new wording.

## Results (leakage-free 5-fold test, 270 sentences)
| Model | Accuracy | Scam recall |
|---|---|---|
| Linear SVM | 92.1% | 98.2% |
| Logistic Regression (used in the app) | 89.4% | 95.9% |

## Run locally
```
pip install -r requirements.txt
streamlit run app_online.py
```

## Limits
- Small, hand-written dataset: results are estimates, not real-world guarantees.
- The score is a model estimate, **not proof of fraud**. The app only warns; it never blocks or ends calls.
- Do not type real OTPs, PINs or card numbers into the app.
- The full local version (not in this online demo) also reads screenshots (Tesseract OCR) and audio (Whisper).

## Data
All sentences are fictional or generic. No real names, phone numbers, OTPs or recordings.
