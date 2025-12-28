# src/scraping.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Educational DS Project)",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
}

SEARCH_URL = "https://www.otodom.pl/pl/wyniki/sprzedaz/mieszkanie/malopolskie/krakow/krakow"


def _fetch_html(url: str, params: Optional[dict] = None) -> str:
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.text


def _extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    tag = soup.select_one("script#__NEXT_DATA__")
    if not tag or not tag.text:
        return None
    return json.loads(tag.text)


def _find_offer_dicts(obj: Any) -> List[Dict[str, Any]]:
    """
    Heurystyka: w __NEXT_DATA__ szukamy list słowników, które wyglądają jak oferty.
    Struktura bywa zmienna, więc robimy wyszukiwanie rekursywne.
    """
    results: List[Dict[str, Any]] = []

    def walk(x: Any):
        if isinstance(x, dict):
            # typowe pola w ofertach (różne warianty)
            keys = set(x.keys())
            if {"price", "area", "rooms", "title", "url", "href"}.intersection(keys):
                results.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    # często jest dużo śmieci — na start zostawiamy, a potem zawęzimy po realnym podglądzie danych
    return results


def _normalize_offer(raw: Dict[str, Any]) -> Dict[str, Any]:
    def pick(*keys):
        for k in keys:
            if k in raw and raw.get(k) not in (None, ""):
                return raw.get(k)
        return None

    return {
        "title": pick("title", "name"),
        "price": pick("price", "totalPrice", "value"),
        "area_m2": pick("area", "areaInSquareMeters", "surface"),
        "rooms": pick("rooms", "numberOfRooms"),
        "url": pick("url", "href", "link"),
        "source": "otodom",
    }


def collect_raw_listings(pages: int = 1, sleep_s: float = 2.0) -> pd.DataFrame:
    """
    Minimalny scraping: pobiera 1..N stron wyników.
    Bez agresywnego pobierania, bez obchodzenia blokad.
    """
    rows: List[Dict[str, Any]] = []

    for page in range(1, pages + 1):
        html = _fetch_html(SEARCH_URL, params={"page": page})
        next_data = _extract_next_data(html)
        if not next_data:
            # Jeśli Otodom nie serwuje __NEXT_DATA__ albo blokuje – kończymy uczciwie.
            raise RuntimeError("Could not find __NEXT_DATA__ in HTML (site changed or access blocked).")

        offer_dicts = _find_offer_dicts(next_data.get("props", {}))
        for o in offer_dicts:
            rows.append(_normalize_offer(o))

        time.sleep(sleep_s)

    df = pd.DataFrame(rows).drop_duplicates()
    return df
