"""
Lexical / structural feature extraction for URL phishing detection.

Only the URL string itself is inspected (no page download, no DNS lookups), so
extraction is fast and safe. The scheme (http/https), a leading "www." and a
trailing "/" are deliberately ignored: phishing sites use HTTPS all the time, so
the scheme carries almost no signal and would only add dataset bias.
"""
import math
import os
import re
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- #
# Word lists
# --------------------------------------------------------------------------- #
SUSPICIOUS_WORDS = (
    "login", "signin", "sign-in", "logon", "verify", "verification", "secure",
    "security", "account", "update", "confirm", "password", "passwd", "banking",
    "bank", "paypal", "webscr", "wallet", "billing", "invoice", "suspend",
    "unlock", "recover", "authenticate", "validate", "helpdesk", "alert",
    "credential", "ebayisapi", "appleid", "apple-id", "cmd=", "dispatch",
)

BRANDS = (
    "paypal", "apple", "google", "microsoft", "amazon", "facebook", "netflix",
    "instagram", "whatsapp", "linkedin", "dropbox", "docusign", "adobe", "fedex",
    "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "icloud",
    "outlook", "office365", "onedrive", "steam", "binance", "coinbase",
    "metamask", "ebay", "yahoo", "verizon", "hdfc", "icici", "paytm", "santander",
)

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "link", "work", "zip",
    "mov", "icu", "cam", "monster", "rest", "buzz", "country", "stream",
    "download", "loan", "review", "men", "win", "bid", "cyou", "sbs", "cfd",
    "support", "live", "date", "racing", "party", "gdn", "kim", "science",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "t.ly", "lnkd.in",
}

FREE_HOSTING = (
    "000webhostapp.com", "weebly.com", "wixsite.com", "blogspot.com", "github.io",
    "netlify.app", "vercel.app", "web.app", "firebaseapp.com", "herokuapp.com",
    "glitch.me", "pages.dev", "workers.dev", "godaddysites.com", "sites.google.com",
    "yolasite.com", "myftpupload.com", "infinityfreeapp.com", "ngrok.io",
    "repl.co", "azurewebsites.net", "r2.dev", "weeblysite.com", "webflow.io",
    "000webhost.com", "byethost.com", "freehostia.com", "hpage.com", "jimdofree.com",
)

# Common two-level public suffixes (small built-in list, avoids an extra dependency)
SECOND_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "edu.au",
    "gov.au", "co.in", "org.in", "net.in", "ac.in", "gov.in", "nic.in", "com.br",
    "com.cn", "co.jp", "or.jp", "ne.jp", "co.za", "com.mx", "co.nz", "com.tr",
    "com.sg", "com.hk", "com.tw", "com.ar", "co.id", "co.kr", "com.my", "com.ph",
    "com.pk", "com.ng", "com.eg", "co.il", "com.ua", "com.ru",
}

TOP_DOMAINS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "top_domains.txt")
_TOP_RANK = None


def _top_rank():
    """Lazy-load {domain: rank} from ml/data/top_domains.txt (popularity list)."""
    global _TOP_RANK
    if _TOP_RANK is None:
        _TOP_RANK = {}
        if os.path.exists(TOP_DOMAINS_PATH):
            with open(TOP_DOMAINS_PATH, encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    d = line.strip().lower()
                    if d:
                        _TOP_RANK.setdefault(d, i)
    return _TOP_RANK


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
_DEC_IP_RE = re.compile(r"^\d{8,10}$")
_DIGIT_RUN_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")

# --------------------------------------------------------------------------- #
# Feature catalogue (order matters: it is the column order of the model input)
# --------------------------------------------------------------------------- #
FEATURE_LABELS = {
    "url_length": "URL length",
    "host_length": "Hostname length",
    "path_length": "Path length",
    "query_length": "Query string length",
    "num_dots_url": "Dots in URL",
    "num_dots_host": "Dots in hostname",
    "num_subdomains": "Number of subdomains",
    "subdomain_length": "Subdomain length",
    "reg_domain_length": "Registered domain length",
    "num_hyphens_host": "Hyphens in hostname",
    "num_hyphens_url": "Hyphens in URL",
    "num_underscores": "Underscores in URL",
    "num_slashes": "Slashes in URL",
    "num_digits_url": "Digits in URL",
    "digit_ratio": "Digit ratio (URL)",
    "letter_ratio": "Letter ratio (URL)",
    "num_digits_host": "Digits in hostname",
    "host_digit_ratio": "Digit ratio (hostname)",
    "num_params": "Query parameters",
    "num_special_chars": "Special characters",
    "has_at": "Contains '@' symbol",
    "has_ip_host": "IP address as host",
    "has_port": "Explicit port number",
    "has_punycode": "Punycode / IDN host",
    "pct_encoded": "Percent-encoded characters",
    "path_depth": "Path depth",
    "tld_length": "TLD length",
    "suspicious_tld": "Suspicious TLD",
    "is_shortener": "URL shortener",
    "free_hosting": "Free hosting / site builder",
    "suspicious_words": "Phishing keywords in URL",
    "suspicious_words_host": "Phishing keywords in hostname",
    "brand_mismatch": "Brand name outside real domain",
    "host_entropy": "Hostname randomness (entropy)",
    "url_entropy": "URL randomness (entropy)",
    "longest_host_label": "Longest hostname label",
    "longest_token": "Longest alphanumeric token",
    "max_digit_run": "Longest run of digits",
    "ext_php": "Points to a .php script",
    "ext_html": "Points to an .htm/.html page",
    "ext_risky": "Points to an executable/archive",
    "embedded_url": "Another URL embedded in path",
    "double_slash_path": "Double slash inside path",
    "host_vowel_ratio": "Vowel ratio (hostname)",
    "in_top_domains": "Domain in popular-sites list",
    "domain_rank_log": "Domain popularity rank (log10)",
}
FEATURE_NAMES = list(FEATURE_LABELS.keys())


def normalize_url(url):
    """Strip whitespace, scheme, leading 'www.' and a trailing '/'."""
    url = (url or "").strip()
    url = _SCHEME_RE.sub("", url)
    if url.lower().startswith("www."):
        url = url[4:]
    if url.endswith("/"):
        url = url[:-1]
    return url


def _entropy(text):
    if not text:
        return 0.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _split(url):
    """Return (host, port_present, path, query) from a scheme-less URL."""
    try:
        parts = urlsplit("//" + url)
        host = (parts.hostname or "").lower()
        has_port = parts.port is not None
        path, query = parts.path or "", parts.query or ""
    except ValueError:                       # malformed port / IPv6 bracket
        host_part = url.split("/", 1)[0].split("?", 1)[0]
        host = host_part.rsplit("@", 1)[-1].lower()
        has_port = ":" in host
        rest = url[len(host_part):]
        path, _, query = rest.partition("?")
    return host, has_port, path, query


def is_ip_host(host):
    return bool(
        _IPV4_RE.match(host) or _HEX_IP_RE.match(host)
        or _DEC_IP_RE.match(host) or ":" in host
    )


def split_domain(host):
    """Return (subdomain, registered_domain, tld) for a hostname."""
    if not host or is_ip_host(host):
        return "", host, ""
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in SECOND_LEVEL_SUFFIXES:
        keep = 3
    else:
        keep = 2
    reg = ".".join(labels[-keep:])
    sub = ".".join(labels[:-keep])
    return sub, reg, labels[-1]


def registered_domain(url):
    """Registered domain of a URL (used to group rows when splitting data)."""
    host, _, _, _ = _split(normalize_url(url))
    return split_domain(host)[1]


def extract_features(url):
    """Return an ordered dict {feature_name: numeric value} for one URL."""
    u = normalize_url(url)
    host, has_port, path, query = _split(u)
    sub, reg, tld = split_domain(host)
    u_low, host_low = u.lower(), host.lower()

    n = max(len(u), 1)
    digits = sum(c.isdigit() for c in u)
    letters = sum(c.isalpha() for c in u)
    host_digits = sum(c.isdigit() for c in host)
    host_letters = [c for c in host if c.isalpha()]
    vowels = sum(c in "aeiou" for c in host_letters)
    ip_host = is_ip_host(host)

    path_low = path.lower()
    ext = path_low.rsplit(".", 1)[-1] if "." in path_low.rsplit("/", 1)[-1] else ""

    reg_label = reg.split(".")[0] if reg else ""
    brand_mismatch = int(
        any(b in u_low and b not in reg_label for b in BRANDS)
    ) if not ip_host else int(any(b in u_low for b in BRANDS))

    tokens = _TOKEN_RE.findall(u)
    digit_runs = _DIGIT_RUN_RE.findall(u)
    host_labels = host.split(".") if host else [""]

    free_host = any(host == h or host.endswith("." + h) for h in FREE_HOSTING)
    rank = None if (ip_host or free_host) else _top_rank().get(reg)

    f = {
        "url_length": len(u),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots_url": u.count("."),
        "num_dots_host": host.count("."),
        "num_subdomains": 0 if ip_host or not sub else sub.count(".") + 1,
        "subdomain_length": len(sub),
        "reg_domain_length": len(reg),
        "num_hyphens_host": host.count("-"),
        "num_hyphens_url": u.count("-"),
        "num_underscores": u.count("_"),
        "num_slashes": u.count("/"),
        "num_digits_url": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "num_digits_host": host_digits,
        "host_digit_ratio": host_digits / max(len(host), 1),
        "num_params": len([p for p in query.split("&") if p]) if query else 0,
        "num_special_chars": sum(u.count(c) for c in "@?&=%#~+$!*,;"),
        "has_at": int("@" in u),
        "has_ip_host": int(ip_host),
        "has_port": int(has_port),
        "has_punycode": int("xn--" in host_low),
        "pct_encoded": len(_PCT_RE.findall(u)),
        "path_depth": len([p for p in path.split("/") if p]),
        "tld_length": len(tld),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(reg in SHORTENERS),
        "free_hosting": int(free_host),
        "suspicious_words": sum(u_low.count(w) for w in SUSPICIOUS_WORDS),
        "suspicious_words_host": sum(host_low.count(w) for w in SUSPICIOUS_WORDS),
        "brand_mismatch": brand_mismatch,
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(u),
        "longest_host_label": max(len(x) for x in host_labels),
        "longest_token": max((len(t) for t in tokens), default=0),
        "max_digit_run": max((len(d) for d in digit_runs), default=0),
        "ext_php": int(ext in ("php", "php3", "php5", "asp", "aspx", "jsp", "cgi")),
        "ext_html": int(ext in ("htm", "html", "shtml")),
        "ext_risky": int(ext in ("exe", "zip", "rar", "apk", "scr", "jar", "msi", "bat", "js")),
        "embedded_url": u_low.count("http", len(host)),
        "double_slash_path": int("//" in path),
        "host_vowel_ratio": vowels / max(len(host_letters), 1),
        "in_top_domains": int(rank is not None),
        "domain_rank_log": math.log10(rank) if rank else 6.0,   # 6.0 = unranked
    }
    return f


def features_to_vector(feature_dict):
    return [float(feature_dict[name]) for name in FEATURE_NAMES]


def format_value(name, value):
    """Human-friendly rendering of a feature value for the UI."""
    if name.startswith(("has_", "is_", "ext_", "suspicious_tld", "free_hosting",
                        "brand_mismatch", "double_slash", "in_top_domains")):
        return "Yes" if value else "No"
    if name == "domain_rank_log":
        return "Unranked" if value >= 5.99 else f"#{int(round(10 ** value)):,}"
    if "ratio" in name:
        return f"{value * 100:.0f}%"
    if "entropy" in name:
        return f"{value:.2f}"
    return f"{int(round(value))}"
