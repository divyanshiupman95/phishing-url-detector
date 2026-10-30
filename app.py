"""
app.py
------
Phishing URL Detector - Streamlit Frontend Application.

Clean, simplified, college-level interface that accepts a URL,
extracts structural features, predicts risk category using the
trained Machine Learning model, and displays detected suspicious indicators.
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Import shared feature extraction and validation logic
from feature_extraction import (
    extract_features,
    get_url_red_flags,
    is_valid_url,
    _normalize_url,
    FEATURE_NAMES
)
from urllib.parse import urlparse

# Known high-reputation domains for transparent educational advisory (NO score override)
HIGH_REPUTATION_DOMAINS = {
    "github.com",
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "paytm.com",
    "microsoft.com",
    "python.org",
    "openai.com",
    "sbi.co.in",
    "hdfcbank.com"
}

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_trained_model(model_path="phishing_model.pkl"):
    """Loads and caches the trained model bundle."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def classify_risk(phishing_prob: float):
    """
    Maps the model's continuous phishing probability to three categories.
    Thresholds:
      - Score < 35%: Likely Safe
      - Score 35% - 70%: Suspicious
      - Score > 70%: Phishing
    """
    if phishing_prob < 0.35:
        return "Likely Safe"
    elif phishing_prob < 0.70:
        return "Suspicious"
    else:
        return "Phishing"


def main():
    # 1. Project Title
    st.title("Phishing URL Detector")

    # 2. Subtitle
    st.caption("Machine Learning Based URL Analysis")
    st.write("")

    # Load model
    bundle = load_trained_model("phishing_model.pkl")
    if bundle is None:
        st.error("Model file ('phishing_model.pkl') not found. Please ensure 'phishing_model.pkl' is in the project folder or run 'python train_experimental_model.py'.")
        return

    model = bundle["model"]

    # 3. One URL input field
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com/login"
    )

    # 4. One Check URL button
    check_btn = st.button("Check URL")

    # 5. Prediction execution
    if check_btn:
        cleaned_url = url_input.strip()
        if not cleaned_url:
            st.warning("Please enter a URL to check.")
            return

        # Input validation: check for valid URL structure and scheme
        if not is_valid_url(cleaned_url):
            st.error("Please enter a valid URL (e.g., https://example.com)")
            return

        # Inform user if scheme was omitted and HTTPS was assumed
        lower_input = cleaned_url.lower()
        if not lower_input.startswith(("http://", "https://")):
            st.info("ℹ️ No URL scheme provided. Assumed **https://** for structural analysis.")

        # Feature extraction using exact shared FEATURE_NAMES order
        feats = extract_features(cleaned_url)
        feats_df = pd.DataFrame([[feats[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # Model prediction
        probs = model.predict_proba(feats_df)[0]
        phishing_prob = float(probs[1])
        category = classify_risk(phishing_prob)

        # Detected red flags
        indicators = get_url_red_flags(cleaned_url)

        st.divider()

        # 5. Predicted Category
        if category == "Likely Safe":
            st.success(f"**Category:** {category}")
        elif category == "Suspicious":
            st.warning(f"**Category:** {category}")
        else:
            st.error(f"**Category:** {category}")

        # 6. Model Phishing Score
        st.write(f"**Phishing Score:** {phishing_prob * 100:.2f}% *(Estimated Model Score)*")

        # 7. Suspicious Indicators List
        st.write("**Detected Suspicious Indicators:**")
        if indicators:
            for item in indicators:
                st.write(f"- {item}")
        else:
            if category in ("Phishing", "Suspicious"):
                st.write(
                    "No individual heuristic red flags triggered. However, the machine learning model "
                    "flagged this URL based on a combination of structural patterns (such as URL length, "
                    "subdomain hierarchy, and lexical composition)."
                )
            else:
                st.write("No suspicious lexical indicators detected.")

        # Optional Exact-Hostname Educational Advisory (Strictly Informational - NO Score Override)
        norm_url = _normalize_url(cleaned_url)
        parsed_norm = urlparse(norm_url)
        raw_host = (parsed_norm.hostname or "").lower()
        base_host = raw_host[4:] if raw_host.startswith("www.") else raw_host

        if base_host in HIGH_REPUTATION_DOMAINS and category in ("Phishing", "Suspicious"):
            st.info(
                f"💡 **Domain Architecture Advisory:** The exact hostname matches a known high-reputation platform (`{base_host}`). "
                f"The model's elevated phishing score ({phishing_prob * 100:.2f}%) is a documented false positive "
                "arising from lexical-only dataset bias (bare domain lacking 'www.' prefix or sensitive keyword collision). "
                "The ML score above is preserved without override. In production security systems, WHOIS domain age "
                "and reputation feeds resolve this without modifying ML model weights."
            )

        st.write("")

        # 8. Brief Security Disclaimer
        st.caption(
            "Disclaimer: This assessment is an automated estimate generated by a machine learning model "
            "based on URL lexical and structural patterns alone. 'Likely Safe' is only an algorithmic estimate "
            "and does not provide a guarantee of safety or security. This application does not fetch or visit the URL."
        )


if __name__ == "__main__":
    main()
