"""
Configuration: default company universe + secrets loading.

On Kaggle, secrets came from `kaggle_secrets.UserSecretsClient`. Outside Kaggle
(local, CI, or any other host) they come from environment variables instead —
set them before running:

    export GEMINI_API_KEY="..."
    export SEC_EDGAR_EMAIL="you@example.com"   # required by SEC EDGAR fair-access policy
"""

import os

# 6 companies, alag-alag sectors se — taaki filing table formats bhi vary karein
# aur system generalize karta hai ye prove ho
DEFAULT_COMPANIES = {
    "NVIDIA": "NVDA",              # semiconductors
    "AMD": "AMD",                  # semiconductors
    "Apple": "AAPL",               # consumer tech
    "Walmart": "WMT",              # retail
    "Johnson & Johnson": "JNJ",    # healthcare
    "JPMorgan Chase": "JPM",       # banking/finance
}

DEFAULT_LLM_MODEL = "gemma-4-31b-it"  # open-weight (Apache 2.0), free tier ~15 RPM / 1M tokens/day


def get_gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it as an environment variable "
            "(or load it from Kaggle Secrets if running on Kaggle)."
        )
    return key


def get_sec_email() -> str:
    email = os.environ.get("SEC_EDGAR_EMAIL")
    if not email:
        raise EnvironmentError(
            "SEC_EDGAR_EMAIL not set. SEC EDGAR's fair-access policy requires a "
            "contact email in the User-Agent header for all requests."
        )
    return email
