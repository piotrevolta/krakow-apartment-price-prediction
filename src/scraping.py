"""
scraping.py

Minimal, polite scraper for collecting apartment listing data from Otodom search results.

Design goals:
- keep logic in src/ (not in notebook)
- avoid aggressive traffic (sleep between pages)
- do not attempt to bypass blocks / protections
- return a "raw but usable" DataFrame for downstream cleaning/EDA

Dependencies:
- requests
- beautifulsoup4
- lxml
- pandas
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class ScrapeConfig:
    base_url: str = "https://www.otodom.pl"
    # Kraków sale apartments results page (you can change later)
    search_url: str = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow"
    user_agent: str = "Mozilla/5.0 (Educational DS project; contact: example@example.com)"
    accept_language: str = "pl-PL,pl;q=0.9,en;q=0.8"
    timeout_s: int = 20
    sleep_s: float = 2.0  # polite delay between pages


CFG = ScrapeConfig()

HEADERS = {
    "User-Agent": CFG.user_agent,
    "Accept-Language": CFG.accept_language,
}


# ----------------------------
# HTTP + parsing helpers
# ----------------------------

def _fetch_html(url: str, params: Optional[dict] = None) -> str:
    r = requests.get(url, headers=HEADERS, params=params, timeout=CFG.timeout_s)
    r.raise_for_status()
    return r.text


def _extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    """
    Otodom uses Next.js. Often the page contains a script tag with JSON state:
    <script id="__NEXT_DATA__"> ... </script>
    """
    soup = BeautifulSoup(html, "lxml")
    tag = soup.select_one("script#__NEXT_DATA__")
    if not tag or not tag.text:
        return None
    try:
        return json.loads(tag.text)
    except json.JSONDecodeError:
        return None


def _walk_find_dicts(obj: Any) -> List[Dict[str, Any]]:
    """
    Recursive scan over JSON for dict-like nodes. We will filter later.
    """
    out: List[Dict[str, Any]] = []

    def walk(x: Any):
        if isinstance(x, dict):
            out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            return d.get(k)
    return None


def _to_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    # sometimes numbers arrive as strings
    if isinstance(x, str):
        try:
            return float(x.replace(",", ".").strip())
        except ValueError:
            return None
    return None


def _normalize_price(price_obj: Any) -> tuple[Optional[int], Optional[str]]:
    """
    Examples you showed:
      {'value': 1145300, 'currency': 'PLN', '__typename': 'Money'}
    Sometimes price may be a number or nested differently.
    """
    if price_obj is None:
        return None, None

    if isinstance(price_obj, dict):
        value = price_obj.get("value")
        curr = price_obj.get("currency")
        if isinstance(value, (int, float)):
            return int(value), curr if isinstance(curr, str) else None
        # fallback if value is string
        num = _to_number(value)
        return (int(num) if num is not None else None), curr if isinstance(curr, str) else None

    if isinstance(price_obj, (int, float)):
        return int(price_obj), None

    # string fallback
    num = _to_number(price_obj)
    return (int(num) if num is not None else None), None


def _normalize_url(url: Any) -> Optional[str]:
    """
    Your sample:
      [lang]/ad/....   (relative path)

    We'll convert:
      "[lang]/ad/xxx" -> "https://www.otodom.pl/pl/ad/xxx"

    (Often it redirects to /pl/oferta/...; that's fine for dedup + later enrichment.)
    """
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()

    if u.startswith("[lang]"):
        u = u.replace("[lang]", "pl", 1)

    # make absolute
    if u.startswith("http://") or u.startswith("https://"):
        return u

    return f"{CFG.base_url}/{u.lstrip('/')}"


def _looks_like_listing(d: Dict[str, Any]) -> bool:
    """
    Heuristic: listing-like dict tends to have title + price + url-ish fields.
    This is intentionally simple for MVP.
    """
    has_title = _pick(d, "title", "name") is not None
    has_price = _pick(d, "price", "totalPrice", "value", "priceFrom") is not None
    has_url = _pick(d, "url", "href", "link") is not None
    return bool(has_title and has_price and has_url)


def _get_attribute(attrs: Any, names: list[str]) -> Any:
    """
    attrs: list of attribute dicts or dict
    names: possible names to match (case-insensitive)
    """
    if not attrs:
        return None

    if isinstance(attrs, dict):
        attrs = attrs.values()

    for a in attrs:
        if not isinstance(a, dict):
            continue

        label = str(a.get("label") or a.get("name") or "").lower()
        key = str(a.get("key") or "").lower()

        for n in names:
            if n in label or n in key:
                return a.get("value")

    return None
    
def _extract_district(d: Dict[str, Any]) -> Optional[str]:
    loc = d.get("location") or {}
    address = loc.get("address") or {}

    for key in ["district", "neighbourhood", "quarter"]:
        if key in address:
            return address.get(key)

    # fallback: czasem jest w path/slug
    slug = address.get("slug")
    if isinstance(slug, str):
        return slug.replace("-", " ").title()

    return None

import re

def _extract_rooms_from_title(title: str) -> Optional[int]:
    if not title:
        return None

    # 3-pokojowe, 2 pokojowe, 4 pokoje
    m = re.search(r"(\d+)\s*[- ]?\s*pokoj", title.lower())
    if m:
        return int(m.group(1))

    return None
    
def _extract_district_from_url(url: str) -> Optional[str]:
    if not url:
        return None

    slug = url.split("/ad/")[-1]
    slug = slug.split("-ul-")[0]   # ulica odcinamy
    slug = slug.replace("-", " ")
    return slug.title()
   district = _extract_district(d)

if district is None:
    district = _extract_district_from_url(url)



def _normalize_listing(d: Dict[str, Any]) -> Dict[str, Any]:
    title = _pick(d, "title", "name")

    price_obj = _pick(d, "price", "totalPrice", "priceFrom", "value")
    price_pln, currency = _normalize_price(price_obj)

    area = _pick(d, "area", "areaInSquareMeters", "surface")
    area_m2 = _to_number(area)

    rooms = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["pokoje", "rooms"]
    )
    rooms = int(_to_number(rooms)) if rooms else None

    # fallback z tytułu
    if rooms is None:
        rooms = _extract_rooms_from_title(title)


    floor = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["piętro", "floor"]
    )
    floor = int(_to_number(floor)) if floor else None
    
    if floor is None:
        floor = _extract_floor_from_title(title)


    elevator_raw = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["winda", "elevator"]
    )
    elevator = 1 if str(elevator_raw).lower() in ("tak", "yes", "true") else 0

    district = _extract_district(d)

    if district is None:
        district = _extract_district_from_url(url)


    url = _normalize_url(_pick(d, "url", "href", "link"))

    return {
        "title": title,
        "price_pln": price_pln,
        "currency": currency,
        "area_m2": area_m2,
        "rooms": rooms,
        "floor": floor,
        "elevator": elevator,
        "district": district,
        "url": url,
        "source": "otodom",
    }



# ----------------------------
# Public API
# ----------------------------

def collect_raw_listings(pages: int = 1, sleep_s: Optional[float] = None) -> pd.DataFrame:
    """
    Collect raw listings from Otodom search results.

    Notes:
    - This function does not bypass protections. If you get blocked or the site structure changes,
      you should stop and switch to an allowed data source.
    - Start small (pages=1..2) and increase carefully.

    Parameters
    ----------
    pages : int
        Number of results pages to fetch (starting from 1).
    sleep_s : float | None
        Delay between requests; defaults to CFG.sleep_s.

    Returns
    -------
    pandas.DataFrame
        Raw-ish dataset with normalized basic fields.
    """
    if pages < 1:
        raise ValueError("pages must be >= 1")

    delay = CFG.sleep_s if sleep_s is None else float(sleep_s)

    rows: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        html = _fetch_html(CFG.search_url, params={"page": page})
        next_data = _extract_next_data(html)

        if not next_data:
            raise RuntimeError(
                "Could not find/parse __NEXT_DATA__. The site may have changed or access is blocked."
            )

        # scan props area first (usually contains page data)
        props = next_data.get("props", {})
        dict_nodes = _walk_find_dicts(props)

        for node in dict_nodes:
            if _looks_like_listing(node):
                rows.append(_normalize_listing(node))

        # polite delay
        time.sleep(delay)

    df = pd.DataFrame(rows)

    # De-duplicate safely (avoid dict hashing issues): use URL if available
    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    return df
