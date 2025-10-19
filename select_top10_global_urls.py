from pathlib import Path
from urllib.parse import urlparse
import unicodedata

from filter_sitemap_to_xlsx import (
    read_robots_disallows,
    build_disallow_regexes,
    extract_sitemap_urls,
    is_disallowed,
    write_minimal_xlsx,
    ROBOTS_PATH,
    SITEMAP_PATH,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_XLSX = ROOT / "Top10_Global_Allowed_URLs.xlsx"


def normalize(text: str) -> str:
    # Lowercase and strip accents for robust keyword matching
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


KEYWORD_WEIGHTS = {
    # FAQs / help
    "faq": 6,
    "questions": 5,
    "aide": 5,
    # Conditions / pricing
    "conditions": 7,
    "tarif": 7,
    "frais": 7,
    "cgu": 7,
    "cgv": 7,
    "mentions-legales": 3,
    # Products / categories
    "produit": 6,
    "produits": 6,
    "offre": 5,
    "offres": 5,
    "compte": 5,
    "carte": 5,
    "cartes": 5,
    "pret": 5,
    "credit": 5,
    "epargne": 5,
    "assurance": 5,
}

NEGATIVE_HINTS = [
    "actualites", "actu", "news", "blog", "le-mag",
    "evenement", "event", "video", "podcast"
]


def score_url(url: str) -> float:
    p = urlparse(url)
    path_norm = normalize(p.path)
    # Depth: number of non-empty segments
    segments = [s for s in path_norm.split('/') if s]
    depth = len(segments)

    # Keyword score
    kw_score = 0
    for k, w in KEYWORD_WEIGHTS.items():
        if k in path_norm:
            kw_score += w

    # Negative hints
    neg = any(h in path_norm for h in NEGATIVE_HINTS)

    # Heuristic final score
    score = kw_score * 100.0
    score -= depth * 10.0
    if neg:
        score -= 50.0
    # Prefer shorter total path length a bit
    score -= len(p.path) * 0.1
    return score


def select_top10(allowed_urls: list[str]) -> list[str]:
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in allowed_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # Sort by score desc
    scored = sorted(unique, key=score_url, reverse=True)
    return scored[:10]


def main() -> None:
    robots_text = ROBOTS_PATH.read_text(encoding='utf-8', errors='ignore')
    disallow_rules = read_robots_disallows(robots_text)
    disallow_regexes = build_disallow_regexes(disallow_rules)

    xml_bytes = SITEMAP_PATH.read_bytes()
    urls = extract_sitemap_urls(xml_bytes)
    allowed = [u for u in urls if not is_disallowed(u, disallow_regexes)]

    top10 = select_top10(allowed)
    write_minimal_xlsx(top10, OUTPUT_XLSX)
    print(f"Wrote {len(top10)} URLs to {OUTPUT_XLSX}")


if __name__ == '__main__':
    main()

