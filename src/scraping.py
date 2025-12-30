"""
scraping.py

Minimal, polite scraper for collecting apartment listing URLs (and address text from result cards)
from Otodom search results.

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

import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class ScrapeConfig:
    base_url: str = "https://www.otodom.pl"
    search_url: str = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow"
    user_agent: str = "Mozilla/5.0 (Educational DS project; contact: example@example.com)"
    accept_language: str = "pl-PL,pl;q=0.9,en;q=0.8"
    timeout_s: int = 20

    # polite delays
    sleep_s: float = 2.0  # between result pages


CFG = ScrapeConfig()

HEADERS = {
    "User-Agent": CFG.user_agent,
    "Accept-Language": CFG.accept_language,
}


# ----------------------------
# HTTP
# ----------------------------

def _fetch_html(url: str, params: Optional[dict] = None) -> str:
    r = requests.get(url, headers=HEADERS, params=params, timeout=CFG.timeout_s)
    r.raise_for_status()
    return r.text


# ----------------------------
# Field extractors (ADD COLUMNS HERE)
# ----------------------------

Extractor = Callable[[Tag], Any]


def _extract_listing_url(card: Tag) -> Optional[str]:
    a = card.select_one('a[href*="/pl/oferta/"]')
    if not a:
        for cand in card.find_all("a", href=True):
            if re.search(r"/pl/oferta/", cand["href"]):
                a = cand
                break
    if not a:
        return None

    href = a.get("href", "").strip()
    if not href:
        return None

    return urljoin(CFG.base_url, href)

# Adres parsing 


def _split_address_parts(card: Tag) -> Optional[List[str]]:
    txt = _extract_address_text(card)
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    return parts or None

def _extract_address_street(card: Tag) -> Optional[str]:
    parts = _split_address_parts(card)
    if not parts:
        return None
    # jeśli są dokładnie 4 elementy: brak ulicy
    if len(parts) == 4:
        return None
    # jeśli jest 5+ elementów: ulica to wszystko przed ostatnimi 4
    if len(parts) >= 5:
        street = ", ".join(parts[0 : len(parts) - 4]).strip()
        return street or None
    # 3 i mniej też traktuj jako brak ulicy
    return None


def _extract_address_subdistrict(card: Tag) -> Optional[str]:
    parts = _split_address_parts(card)
    # przy 4 elementach: subdistrict = pierwszy segment
    if parts and len(parts) == 4:
        return parts[0]
    return parts[-4] if parts and len(parts) >= 4 else None


def _extract_address_district(card: Tag) -> Optional[str]:
    parts = _split_address_parts(card)
    # przy 4 elementach: district = drugi segment
    if parts and len(parts) == 4:
        return parts[1]
    return parts[-3] if parts and len(parts) >= 3 else None


def _extract_address_city(card: Tag) -> Optional[str]:
    parts = _split_address_parts(card)
    return parts[-2] if parts and len(parts) >= 2 else None


def _extract_address_voivodeship(card: Tag) -> Optional[str]:
    parts = _split_address_parts(card)
    return parts[-1] if parts and len(parts) >= 1 else None


def _extract_address_text(card: Tag) -> Optional[str]:
    # dokładnie to, co masz w HTML (Address component)
    p = card.select_one('p[data-sentry-component="Address"]')
    if not p:
        return None

    text = p.get_text(" ", strip=True)
    return text or None

# ========================== Price

def _extract_price_text(card: Tag) -> Optional[str]:
    price = card.select_one(
        'div[data-sentry-component="CustomizedPrice"] span[data-sentry-element="MainPrice"]'
    )
    if not price:
        return None
    text = price.get_text(" ", strip=True)
    return text or None


def _extract_price_per_m2_text(card: Tag) -> Optional[str]:
    # drugi span w CustomizedPrice (bez parsowania na liczbę)
    wrapper = card.select_one('div[data-sentry-component="CustomizedPrice"]')
    if not wrapper:
        return None

    spans = wrapper.find_all("span")
    if not spans or len(spans) < 2:
        return None

    text = spans[1].get_text(" ", strip=True)
    return text or None


FIELD_EXTRACTORS: Dict[str, Extractor] = {
    "listing_url": _extract_listing_url,
    "address_text": _extract_address_text,
    "address_street": _extract_address_street,
    "address_subdistrict": _extract_address_subdistrict,
    "address_district": _extract_address_district,
    "address_city": _extract_address_city,
    "address_voivodeship": _extract_address_voivodeship,
    "price_text": _extract_price_text,
    "price_per_m2_text": _extract_price_per_m2_text,
}


# ----------------------------
# Parsing
# ----------------------------
def _find_result_cards(soup: BeautifulSoup) -> List[Tag]:
    out: List[Tag] = []
    for art in soup.find_all("article"):
        if art.select_one('a[href*="/pl/oferta/"]') and art.select_one('p[data-sentry-component="Address"]'):
            out.append(art)
    return out



def parse_results_page(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    cards = _find_result_cards(soup)

    rows: List[Dict[str, Any]] = []
    for card in cards:
        row: Dict[str, Any] = {}
        for col_name, extractor in FIELD_EXTRACTORS.items():
            try:
                row[col_name] = extractor(card)
            except Exception:
                row[col_name] = None
        rows.append(row)

    return rows


# ----------------------------
# Public API
# ----------------------------

def scrape_search(max_pages: int = 1) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []

    for page in range(1, max_pages + 1):
        params = None if page == 1 else {"page": page}
        html = _fetch_html(CFG.search_url, params=params)
        all_rows.extend(parse_results_page(html))

        if page < max_pages:
            time.sleep(CFG.sleep_s)

    return pd.DataFrame(all_rows)


def collect_raw_listings(max_pages: int = 1) -> pd.DataFrame:
    return scrape_search(max_pages=max_pages)

