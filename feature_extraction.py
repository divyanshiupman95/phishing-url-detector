"""
feature_extraction.py
---------------------
Yeh module URLs se Machine Learning features aur human-readable 'Red Flags'
(suspicious patterns) extract karta hai.

Dono training (`model_training.py`) aur live web application (`app.py`)
isi same module ko use karenge taaki 'Training-Serving Skew' na ho.
"""

import re
import ipaddress
from urllib.parse import urlparse
import pandas as pd

# List of suspicious keywords frequently used in phishing attacks
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "bank", "secure", "kyc",
    "account", "signin", "confirm", "password", "banking", "wallet"
]

# Fixed list of numerical feature names in exact order
# Is order ko change nahi karna hai, taaki model training aur prediction ka order same rahe.
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "has_at_symbol",
    "num_digits",
    "digit_ratio",
    "is_https",
    "has_ip_address",
    "num_subdomains",
    "num_suspicious_words",
    "has_double_slash_redirect",
    "url_depth"
]


def is_valid_url(url: str) -> bool:
    """
    Validates whether the input string is a valid web URL.
    
    Validation criteria:
    - Must be a non-empty string without whitespace.
    - Scheme (if explicitly specified) must be 'http' or 'https' (case-insensitive).
      Rejects non-web schemes like 'javascript:', 'data:', 'mailto:', 'ftp:'.
    - Hostname must be a valid IPv4/IPv6 address or a valid domain name.
    - Rejects 'localhost' and single-word local network hosts.
    - Domain name must have at least two labels (domain + TLD) with a valid TLD
      (at least 2 letters, e.g. .com, .org, .co.in).
    - Labels must conform to RFC 1035 / RFC 1123 naming rules.
    """
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url or any(c.isspace() for c in url):
        return False

    # Check for explicit non-HTTP/HTTPS schemes (case-insensitive)
    if ":" in url:
        prefix = url.split(":", 1)[0].strip().lower()
        if re.match(r"^[a-z][a-z0-9+\-.]*$", prefix):
            if prefix not in ("http", "https"):
                return False

    # Normalize temporary URL for parsing
    lower_url = url.lower()
    if lower_url.startswith(("http://", "https://")):
        scheme, remainder = url.split("://", 1)
        temp_url = f"{scheme.lower()}://{remainder}"
    else:
        temp_url = "https://" + url

    try:
        parsed = urlparse(temp_url)
    except Exception:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower()

    # Reject localhost explicitly
    if hostname == "localhost":
        return False

    # Check if hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Domain name structural validation
    if "." not in hostname:
        return False

    labels = hostname.split(".")
    if len(labels) < 2:
        return False

    # TLD must be alphabetic and at least 2 characters (e.g. .com, .org, .in)
    tld = labels[-1]
    if not re.match(r"^[a-z]{2,63}$", tld):
        return False

    # RFC 1035 / RFC 1123 label rules
    label_regex = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?$")
    for label in labels:
        if not label or not label_regex.match(label):
            return False

    return True


def _normalize_url(url: str) -> str:
    """
    Standardizes URL string for consistent feature extraction.
    
    Normalization rules:
    1. Strip leading and trailing whitespace.
    2. Lowercase the scheme and hostname.
    3. If no scheme is provided, assume HTTPS (e.g. 'google.com' -> 'https://google.com').
    4. Remove client-side fragment identifiers (e.g. '#top').
    5. Remove redundant root trailing slash (e.g. 'https://example.com/' -> 'https://example.com').
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    # Parse and extract scheme (case-insensitive)
    lower_url = url.lower()
    if lower_url.startswith(("http://", "https://")):
        scheme, remainder = url.split("://", 1)
        scheme = scheme.lower()
        temp_url = f"{scheme}://{remainder}"
    else:
        # Scheme-less: assume https://
        scheme = "https"
        temp_url = f"https://{url}"

    try:
        parsed = urlparse(temp_url)
        netloc = parsed.netloc.lower()
        path = parsed.path

        # RFC 3986 Root Trailing Slash Normalization:
        # If path is just "/" and there is no query, remove the root slash
        if path == "/":
            path = ""

        # Reconstruct normalized URL (fragments like #top are stripped)
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"

        return normalized
    except Exception:
        return url


def _get_parsed_url(url: str):
    """
    URL ko parse karta hai.
    Agar URL me 'http://' ya 'https://' nahi hai, toh temporarily add karta hai
    taaki urllib hostname aur path sahi se extract kar sake.
    """
    normalized = _normalize_url(url)
    if not normalized.startswith(("http://", "https://")):
        normalized = "http://" + normalized
    return urlparse(normalized)


def extract_features(url: str) -> dict:
    """
    Ek single URL leta hai aur ek dictionary return karta hai jisme
    har ek feature ka numerical value hota hai.
    
    Parameters:
        url (str): The URL string to inspect
        
    Returns:
        dict: Feature name -> numerical value mapping
    """
    url_clean = _normalize_url(url)
    parsed = _get_parsed_url(url_clean)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # 1. URL Length: Phishing URLs aksar parameters aur tokens chhipane ke liye lambi hoti hain
    url_length = len(url_clean)

    # 2. Domain Length: Domain name ki length
    domain_length = len(hostname)

    # 3. Dots count: Multiple subdomains aur spoofing me dots zyada hote hain
    num_dots = url_clean.count(".")

    # 4. Hyphens count: Phishers legit brands copy karne ke liye hyphens use karte hain (e.g. paypal-security)
    num_hyphens = url_clean.count("-")

    # 5. Presence of '@' symbol: Browser '@' se pehle wale hisse ko ignore kar deta hai
    has_at_symbol = 1 if "@" in url_clean else 0

    # 6. Digits count: Random numbers aur IDs URL me kitne hain
    num_digits = sum(c.isdigit() for c in url_clean)

    # 7. Digit ratio: Total length ke muqable digits ka percentage
    digit_ratio = (num_digits / url_length) if url_length > 0 else 0.0

    # 8. HTTPS Check: Legit sites usually HTTPS use karti hain (1 = Yes, 0 = No)
    is_https = 1 if url_clean.lower().startswith("https://") else 0

    # 9. IP Address Check: Kya hostname ek raw IP address hai? (e.g. http://192.168.1.1/login)
    has_ip = 0
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # 10. Subdomains count: Hostname ke parts minus domain aur TLD
    domain_parts = hostname.split(".")
    num_subdomains = max(0, len(domain_parts) - 2) if len(domain_parts) > 1 else 0

    # 11. Suspicious keywords count: Sensitive words jaise login, verify, bank, kyc etc.
    url_lower = url_clean.lower()
    num_suspicious_words = sum(1 for word in SUSPICIOUS_KEYWORDS if word in url_lower)

    # 12. Double slash redirect check: Path me '//' hona redirect trick ho sakta hai
    has_double_slash = 1 if "//" in path else 0

    # 13. URL Depth: Path me directory slashes kitne hain (kitna deep link hai)
    url_depth = path.count("/")

    # Features ko defined fixed order me dictionary me pack karna
    features = {
        "url_length": float(url_length),
        "domain_length": float(domain_length),
        "num_dots": float(num_dots),
        "num_hyphens": float(num_hyphens),
        "has_at_symbol": float(has_at_symbol),
        "num_digits": float(num_digits),
        "digit_ratio": float(round(digit_ratio, 4)),
        "is_https": float(is_https),
        "has_ip_address": float(has_ip),
        "num_subdomains": float(num_subdomains),
        "num_suspicious_words": float(num_suspicious_words),
        "has_double_slash_redirect": float(has_double_slash),
        "url_depth": float(url_depth)
    }

    return features


def extract_features_list(url: str) -> list:
    """Single URL ke features ko list format me deta hai (Model prediction ke liye)."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


def extract_features_df(urls) -> pd.DataFrame:
    """
    Multiple URLs (list ya pandas Series) se DataFrame format me features banata hai.
    Yeh function training script ke dauran use hota hai.
    """
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def get_url_red_flags(url: str) -> list:
    """
    URL me kaun-kaun se suspicious patterns (Red Flags) paye gaye,
    iski human-readable list return karta hai taaki user aur professor samajh sakein.
    """
    red_flags = []
    url_clean = _normalize_url(url)
    parsed = _get_parsed_url(url_clean)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    url_lower = url_clean.lower()

    # Rule 1: IP address as hostname
    try:
        ipaddress.ip_address(hostname)
        red_flags.append("Hostname is an IP address instead of a domain name (often hides identity).")
    except ValueError:
        pass

    # Rule 2: '@' Symbol
    if "@" in url_clean:
        red_flags.append("Contains '@' symbol (browsers ignore text before '@', often used to mislead users).")

    # Rule 3: Missing HTTPS
    if not url_lower.startswith("https://"):
        red_flags.append("Does not use HTTPS encryption (insecure HTTP connection).")

    # Rule 4: Suspicious Keywords
    found_words = [w for w in SUSPICIOUS_KEYWORDS if w in url_lower]
    if found_words:
        red_flags.append(f"Contains sensitive/suspicious keyword(s): {', '.join(found_words)}.")

    # Rule 5: Excessive dots
    if url_clean.count(".") >= 4:
        red_flags.append(f"High number of dots ({url_clean.count('.')}) detected (potential subdomain spoofing).")

    # Rule 6: Excessive hyphens
    if url_clean.count("-") >= 3:
        red_flags.append(f"Multiple hyphens ({url_clean.count('-')}) detected in URL (common in fake brand names).")

    # Rule 7: Unusually long URL
    if len(url_clean) > 75:
        red_flags.append(f"Unusually long URL ({len(url_clean)} characters; legit URLs are typically shorter).")

    # Rule 8: High digit ratio
    digits = sum(c.isdigit() for c in url_clean)
    if len(url_clean) > 0 and (digits / len(url_clean)) > 0.20:
        red_flags.append(f"High proportion of numbers ({round((digits/len(url_clean))*100)}% digits in URL).")

    # Rule 9: Double slash in path
    if "//" in path:
        red_flags.append("Contains '//' within the path (redirect evasion pattern).")

    # Rule 10: Excessive subdomains
    domain_parts = hostname.split(".")
    subdomains = max(0, len(domain_parts) - 2) if len(domain_parts) > 1 else 0
    if subdomains >= 2:
        red_flags.append(f"Multiple subdomains ({subdomains}) detected.")

    return red_flags


if __name__ == "__main__":
    # Test example
    sample_safe = "https://www.google.com"
    sample_phish = "http://192.168.1.1/secure-bank-login/verify.php?account=123"

    print("--- Safe URL Test ---")
    print(f"URL: {sample_safe}")
    print("Features:", extract_features(sample_safe))
    print("Red Flags:", get_url_red_flags(sample_safe))

    print("\n--- Phishing URL Test ---")
    print(f"URL: {sample_phish}")
    print("Features:", extract_features(sample_phish))
    print("Red Flags:", get_url_red_flags(sample_phish))
