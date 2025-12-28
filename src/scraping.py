"""
scraping.py

Minimal, polite scraper for collecting apartment listing data from Otodom search results.

Design goals:
- keep logic in src/ (not in notebook)
- avoid aggressive traffic (sleep between pages)
- do not attempt to bypass blocks / protections
- return a "raw but usable" DataFrame for downstream cleaning/EDA
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
    search_url: str = (
        "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow"
    )
    user_agent: str = "Mozilla/5.0 (Educational DS project)"
    accept_language: str = "pl-PL,pl;q=0.9,en;q=0.8"
    timeout_s: int = 20
    sleep_s: float = 2.0


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
    soup = BeautifulSoup(html, "lxml")
    tag = soup.select_one("script#__NEXT_DATA__")
    if not tag or not tag.text:
        return None
    try:
        return json.loads(tag.text)
    except json.JSONDecodeError:
        return None


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


# ----------------------------
# Extract listings (KEY FIX)
# ----------------------------

def _extract_listings_from_props(props: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find the actual list of listing dicts inside Next.js props.
    Prefer lists of dicts containing title+url+price.
    """
    candidates: List[List[Dict[str, Any]]] = []

    def walk(x: Any):
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                sample = x[0]
                if (
                    any(k in sample for k in ("url", "href", "link"))
                    and any(k in sample for k in ("title", "name"))
                    and any(k in sample for k in ("price", "totalPrice", "priceFrom", "value"))
                ):
                    candidates.append(x)
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(props)

    if not candidates:
        return []

    return max(candidates, key=len)


# ----------------------------
# Normalizers / extractors
# ----------------------------

def _normalize_price(price_obj: Any) -> tuple[Optional[int], Optional[str]]:
    if price_obj is None:
        return None, None

    if isinstance(price_obj, dict):
        val = price_obj.get("value")
        cur = price_obj.get("currency")
        if isinstance(val, (int, float)):
            return int(val), cur if isinstance(cur, str) else None
        num = _to_number(val)
        return (int(num) if num is not None else None), cur if isinstance(cur, str) else None

    if isinstance(price_obj, (int, float)):
        return int(price_obj), None

    num = _to_number(price_obj)
    return (int(num) if num is not None else None), None


def _normalize_url(url: Any) -> Optional[str]:
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()
    u = u.replace("/hpr/", "/").replace("[lang]", "pl")
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return f"{CFG.base_url}/{u.lstrip('/')}"


def _get_attribute(attrs: Any, names: List[str]) -> Any:
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


def _extract_rooms_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    m = re.search(r"(\d+)\s*[- ]?\s*pokoj", title.lower())
    return int(m.group(1)) if m else None


def _extract_floor_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    t = title.lower()
    if "parter" in t:
        return 0
    m = re.search(r"(\d+)\s*pi[eę]tr", t)
    return int(m.group(1)) if m else None


def _extract_elevator_from_title(title: Optional[str]) -> Optional[int]:
    if not title:
        return None
    t = title.lower()
    if "bez windy" in t:
        return 0
    if "winda" in t:
        return 1
    return None


def _extract_location_text(d: Dict[str, Any]) -> Optional[str]:
    loc = d.get("location") or {}
    if isinstance(loc, dict):
        for k in ("label", "name", "locationLabel"):
            v = loc.get(k)
            if isinstance(v, str) and "," in v:
                return v.strip()

        address = loc.get("address") or {}
        if isinstance(address, dict):
            parts = []
            for key in ("neighbourhood", "district", "city", "province", "region"):
                v = address.get(key)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            if parts:
                return ", ".join(parts)

    return None


def _split_location(location_text: Optional[str]) -> Dict[str, Optional[str]]:
    if not location_text:
        return {"neighbourhood": None, "district": None, "city": None, "province": None}

    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    province = parts[-1] if len(parts) >= 1 else None
    city = parts[-2] if len(parts) >= 2 else None
    district = parts[-3] if len(parts) >= 3 else None
    neighbourhood = ", ".join(parts[:-3]).strip() if len(parts) >= 4 else None

    return {
        "neighbourhood": neighbourhood or None,
        "district": district or None,
        "city": city or None,
        "province": province or None,
    }


# ----------------------------
# Listing normalization
# ----------------------------

def _normalize_listing(d: Dict[str, Any]) -> Dict[str, Any]:
    title = _pick(d, "title", "name")

    price_obj = _pick(d, "price", "totalPrice", "priceFrom", "value")
    price_pln, currency = _normalize_price(price_obj)

    area = _pick(d, "area", "areaInSquareMeters", "surface")
    area_m2 = _to_number(area)

    rooms = _get_attribute(d.get("attributes") or d.get("characteristics"), ["pokoje", "rooms"])
    rooms = int(_to_number(rooms)) if rooms else _extract_rooms_from_title(title)

    floor = _get_attribute(d.get("attributes") or d.get("characteristics"), ["piętro", "floor"])
    floor = int(_to_number(floor)) if floor else _extract_floor_from_title(title)

    elevator_raw = _get_attribute(d.get("attributes") or d.get("characteristics"), ["winda", "elevator"])
    if elevator_raw is None or str(elevator_raw).strip() == "":
        elevator = _extract_elevator_from_title(title)
    else:
        elevator = 1 if str(elevator_raw).lower() in ("tak", "yes", "true") else 0

    url = _normalize_url(_pick(d, "url", "href", "link"))

    location_text = _extract_location_text(d)
    loc_parts = _split_location(location_text)

    return {
        "title": title,
        "price_pln": price_pln,
        "currency": currency,
        "area_m2": area_m2,
        "rooms": rooms,
        "floor": floor,
        "elevator": elevator,
        "location_text": location_text,
        "neighbourhood": loc_parts["neighbourhood"],
        "district": loc_parts["district"],
        "city": loc_parts["city"],
        "province": loc_parts["province"],
        "url": url,
        "source": "otodom",
    }


# ----------------------------
# Public API
# ----------------------------

def collect_raw_listings(pages: int = 1, sleep_s: Optional[float] = None) -> pd.DataFrame:
    if pages < 1:
        raise ValueError("pages must be >= 1")

    delay = CFG.sleep_s if sleep_s is None else float(sleep_s)
    rows: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        html = _fetch_html(CFG.search_url, params={"page": page})
        data = _extract_next_data(html)
        if not data:
            raise RuntimeError("Could not parse __NEXT_DATA__")

        props = data.get("props", {})
        listings = _extract_listings_from_props(props)

        for node in listings:
            rows.append(_normalize_listing(node))

        time.sleep(delay)

    df = pd.DataFrame(rows)
    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset=["url"])

    return df
