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
import re
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
        num = _to_number(value)
        return (int(num) if num is not None else None), curr if isinstance(curr, str) else None

    if isinstance(price_obj, (int, float)):
        return int(price_obj), None

    num = _to_number(price_obj)
    return (int(num) if num is not None else None), None


def _normalize_url(url: Any) -> Optional[str]:
    """
    Your sample:
      [lang]/ad/....   (relative path)

    We'll convert:
      "[lang]/ad/xxx" -> "https://www.otodom.pl/pl/ad/xxx"
    """
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()

    if u.startswith("[lang]"):
        u = u.replace("[lang]", "pl", 1)

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
            val = address.get(key)
            return val if isinstance(val, str) else None

    slug = address.get("slug")
    if isinstance(slug, str):
        return slug.replace("-", " ").title()

    return None
    
def _extract_location_text(d: Dict[str, Any]) -> Optional[str]:
    """
    Tries to extract a human-readable location string like:
    'Bonarka, Podgórze, Kraków, małopolskie'
    """
    # common direct labels (varies)
    for k in ("locationLabel", "location", "address", "geoLabel"):
        v = d.get(k)
        if isinstance(v, str) and "," in v:
            return v.strip()

    # nested: location -> address -> (labels / parts)
    loc = d.get("location") or {}
    if isinstance(loc, dict):
        # sometimes location has a label
        for k in ("label", "name", "locationLabel"):
            v = loc.get(k)
            if isinstance(v, str) and "," in v:
                return v.strip()

        address = loc.get("address") or {}
        if isinstance(address, dict):
            # sometimes address has a label
            for k in ("label", "fullAddress", "full", "name"):
                v = address.get(k)
                if isinstance(v, str) and "," in v:
                    return v.strip()

            # build from parts if present
            parts = []
            for key in ("neighbourhood", "district", "city", "province", "region"):
                v = address.get(key)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

            if parts:
                return ", ".join(parts)

    return None


def _split_location(location_text: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Split 'Bonarka, Podgórze, Kraków, małopolskie' into parts.
    Works even if some parts are missing.
    """
    if not location_text:
        return {"neighbourhood": None, "district": None, "city": None, "province": None}

    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    # heuristic mapping from right side (most stable):
    province = parts[-1] if len(parts) >= 1 else None
    city = parts[-2] if len(parts) >= 2 else None
    district = parts[-3] if len(parts) >= 3 else None
    neighbourhood = ", ".join(parts[:-3]).strip() if len(parts) >= 4 else (parts[0] if len(parts) == 3 else None)

    return {
        "neighbourhood": neighbourhood or None,
        "district": district or None,
        "city": city or None,
        "province": province or None,
    }


def _extract_rooms_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    m = re.search(r"(\d+)\s*[- ]?\s*pokoj", title.lower())
    return int(m.group(1)) if m else None


def _extract_floor_from_title(title: Optional[str]) -> Optional[int]:
    """
    Simple heuristic:
    - 'parter' -> 0
    - 'X piętro' -> X
    """
    if not title:
        return None

    t = title.lower()
    if "parter" in t:
        return 0

    m = re.search(r"(\d+)\s*pi[eę]tr", t)
    return int(m.group(1)) if m else None


def _extract_district_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    # Example: https://www.otodom.pl/pl/ad/ruczaj-os-europejskie-ul-czerwone-maki...
    slug = url.split("/ad/")[-1]
    slug = slug.split("-ul-")[0]  # cut street part if present
    slug = slug.replace("-", " ").strip()

    return slug.title() if slug else None


def _normalize_listing(d: Dict[str, Any]) -> Dict[str, Any]:
    title = _pick(d, "title", "name")

    price_obj = _pick(d, "price", "totalPrice", "priceFrom", "value")
    price_pln, currency = _normalize_price(price_obj)

    area = _pick(d, "area", "areaInSquareMeters", "surface")
    area_m2 = _to_number(area)

    rooms = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["pokoje", "rooms"],
    )
    rooms = int(_to_number(rooms)) if rooms else None
    if rooms is None:
        rooms = _extract_rooms_from_title(title)

    floor = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["piętro", "floor"],
    )
    floor = int(_to_number(floor)) if floor else None
    if floor is None:
        floor = _extract_floor_from_title(title)

    elevator_raw = _get_attribute(
        d.get("attributes") or d.get("characteristics"),
        ["winda", "elevator"],
    )
    # Important: unknown != no elevator → keep None when missing
    if elevator_raw is None or str(elevator_raw).strip() == "":
        elevator = None
    else:
        elevator = 1 if str(elevator_raw).lower() in ("tak", "yes", "true") else 0

    url = _normalize_url(_pick(d, "url", "href", "link"))

    location_text = _extract_location_text(d)
    loc_parts = _split_location(location_text)

    # district bierz z JSON (części lokalizacji) jako priorytet
    district = _extract_district(d) or loc_parts["district"]
    if district is None:
        # dopiero na końcu fallback z URL, ale lepiej go już NIE używać jako "district"
        district = None


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
        "location_text": location_text,
        "neighbourhood": loc_parts["neighbourhood"],
        "district": district,
        "city": loc_parts["city"],
        "province": loc_parts["province"],
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

        props = next_data.get("props", {})
        dict_nodes = _walk_find_dicts(props)

        for node in dict_nodes:
            if _looks_like_listing(node):
                rows.append(_normalize_listing(node))

        time.sleep(delay)

    df = pd.DataFrame(rows)

    # De-duplicate safely: use URL if available
    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    return df
