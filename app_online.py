import os
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="ByteBarrier", page_icon="🛡️")


@st.cache_resource
def load_model():
    """Train the scam model from the Excel dataset when the app starts (takes about a second)."""
    path = "dataset/scam_sentences.xlsx"
    if not os.path.exists(path):
        path = "scam_sentences.xlsx"
    df = (pd.read_excel(path)
            .dropna(subset=["Text", "Label"])
            .drop_duplicates(subset="Text"))
    X = df["Text"].astype(str).str.lower()
    y = (df["Label"] == "Scam").astype(int)          # 1 = Scam, 0 = Normal
    model = Pipeline([
        ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
        ("c", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X, y)
    return model


model = load_model()

# ---------- Advice in 3 languages ----------
ADVICE = {
    "English": {
        "high": "This may be a scam. Do NOT share your OTP, PIN or bank details. Please call your son or a trusted person now.",
        "mid": "This looks suspicious. Do not share OTP or banking details. Consider asking a family member.",
        "low": "No strong scam signs found. Stay careful and never share your OTP.",
    },
    "தமிழ்": {
        "high": "இது மோசடியாக இருக்கலாம். OTP, PIN அல்லது வங்கி விவரங்களை யாருடனும் பகிர வேண்டாம். உடனே உங்கள் மகன் அல்லது நம்பிக்கையான நபரை அழைக்கவும்.",
        "mid": "இது சந்தேகத்திற்குரியது. OTP அல்லது வங்கி விவரங்களை பகிர வேண்டாம். குடும்பத்தினரிடம் கேளுங்கள்.",
        "low": "பெரிய ஆபத்து அறிகுறிகள் இல்லை. ஆனாலும் OTP-ஐ யாருடனும் பகிர வேண்டாம்.",
    },
    "हिन्दी": {
        "high": "यह धोखाधड़ी हो सकती है। OTP, PIN या बैंक की जानकारी किसी को न बताएं। कृपया अभी अपने बेटे या किसी भरोसेमंद व्यक्ति को फोन करें।",
        "mid": "यह संदिग्ध लगता है। OTP या बैंक की जानकारी न दें। परिवार के किसी सदस्य से पूछें।",
        "low": "कोई बड़ा खतरा नहीं दिखा, फिर भी OTP किसी को न बताएं।",
    },
}

# ---------- Key-points layer ----------
SIGNALS = {
    "Asks for OTP / PIN / CVV": (40, ["otp", "pin", "cvv", "password", "ஓடிபி", "ஓ.டி.பி"]),
    "Asks for bank / KYC / card details": (25, ["kyc", "account number", "card number", "bank account",
                                                "debit card", "credit card", "aadhaar", "ஆதார்", "வங்கி கணக்கு"]),
    "Uses urgency": (20, ["urgent", "urgently", "immediately", "right now", "within 24 hours",
                          "today only", "udane", "உடனே", "तुरंत"]),
    "Uses threats": (25, ["arrest", "police", "blocked", "block", "suspended", "legal action",
                          "case", "cbi", "customs", "digital arrest", "கைது", "போலீஸ்", "பிளாக்", "गिरफ्तार"]),
    "Asks for money / fee": (25, ["transfer", "send money", "pay", "fine", "processing fee", "delivery charge",
                                  "registration fee", "clearance fee", "clearance charge",
                                  "refund", "பணம்", "அபராதம்"]),
    "Prize / lottery bait": (25, ["lottery", "you won", "prize", "lucky draw", "free iphone", "you are selected",
                                  "you have been selected", "லாட்டரி", "பரிசு"]),
    "Asks to install an app or keep it secret": (45, ["install this app", "download this app", "control your screen",
                                                      "share your screen", "anydesk", "teamviewer",
                                                      "do not tell anyone", "don't tell anyone",
                                                      "keep this secret", "yaarukum solladheenga"]),
    "Asks to press a key": (15, ["press 1", "press 9", "press any key"]),
}

WARNING_PHRASES = ["never share", "do not share", "don't share", "not share your otp",
                   "பகிர வேண்டாம்", "share panna vendaam"]

def has_word(text, w):
    if w.isascii():
        return re.search(r"\b" + re.escape(w) + r"\b", text) is not None
    return w in text

def rule_check(text):
    text = text.lower()
    is_warning = any(p in text for p in WARNING_PHRASES)
    found, total = [], 0
    for reason, (weight, words) in SIGNALS.items():
        if is_warning and reason.startswith("Asks for OTP"):
            continue
        if any(has_word(text, w) for w in words):
            found.append(reason)
            total += weight
    # screenshots of SMS / WhatsApp often contain links
    if re.search(r"https?://|www\.|bit\.ly|tinyurl|click here|click the link", text):
        found.append("Contains a link")
        total += 20
    if len(found) >= 2:
        total += 10
    return min(total, 100), found

def phone_digits(number):
    digits = re.sub(r"\D", "", number)
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10:
        digits = "91" + digits          # assume India if no country code
    return digits


def show_trusted(trusted):
    if trusted.strip():
        num = phone_digits(trusted)
        st.markdown(f"## 📞 Call this number now: {trusted.strip()}")
        st.markdown(f"[📞 Call on a phone](tel:+{num})  \n"
                    f"[💬 Message on WhatsApp](https://wa.me/{num})")
    else:
        st.info("Add a trusted person's number above so you know who to call.")


# ---------- Shared result display ----------
def analyze(text, language, trusted, key):
    ml_score = round(model.predict_proba([text.lower()])[0][1] * 100)
    rule_score, reasons = rule_check(text)
    score = max(rule_score, round(ml_score * 0.7))

    st.metric("Risk score", f"{score} / 100")

    if score >= 70:
        st.error("🚨 HIGH RISK\n\n" + ADVICE[language]["high"])
        show_trusted(trusted)
    elif score >= 40:
        st.warning("⚠️ SUSPICIOUS\n\n" + ADVICE[language]["mid"])
        show_trusted(trusted)
    else:
        st.success("✅ LOW RISK\n\n" + ADVICE[language]["low"])

    if reasons:
        st.write("**Why this was flagged:**")
        for r in reasons:
            st.write("- " + r)

# ---------- Page ----------
st.title("🛡️ ByteBarrier – Scam Checker")
st.caption("Online demo: checks typed or pasted text in English, Tamil and Tanglish.")

language = st.selectbox("Language / மொழி / भाषा", list(ADVICE.keys()))
trusted = st.text_input("Trusted person's phone number (son / daughter / friend)")
if trusted.strip():
    _n = phone_digits(trusted)
    st.markdown(f"🆘 **Emergency:** [📞 Call {trusted.strip()}](tel:+{_n}) &nbsp;|&nbsp; [💬 WhatsApp](https://wa.me/{_n})")

text = st.text_area("Paste or type what the caller or message said:", height=150)
if st.button("Check text"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        analyze(text, language, trusted, "text")

st.caption("This is a prototype estimate, not proof of fraud. Never share your OTP or PIN. "
           "Do not type real OTPs, PINs or card numbers here. Messages are not saved by this app.")
