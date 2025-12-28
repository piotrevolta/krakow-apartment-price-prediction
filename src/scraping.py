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

    # polite delays
    sleep_s: float = 2.0         # between result pages
    detail_sleep_s: float = 1.2  # between detail pages

    # limit detail fetches (None = all)
    max_listings_details: Optional[int] = None


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
    Examples:
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
            n = n.lower()
            if n in label or n in key:
                return a.get("value")

    return None


def _extract_district(d: Dict[str, Any]) -> Optional[str]:
    """
    Prefer structured location.address.district/neighbourhood/quarter when present.
    """
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
    """
    Heuristic only (NOT a real district).
    Keep as district_guess for debug, do not overwrite district from structured data.
    """
    if not url:
        return None

    slug = url.split("/ad/")[-1]
    slug = slug.split("-ul-")[0]
    slug = slug.replace("-", " ").strip()

    if not slug:
        return None

    return slug.title()


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

    # IMPORTANT:
    # district = ONLY from structured data (no guessing from URL/title)
    district = _extract_district(d)
    district_guess = _extract_district_from_url(url) if district is None else None

    return {
        "title": title,
        "price_pln": price_pln,
        "currency": currency,
        "area_m2": area_m2,
        "rooms": rooms,
        "floor": floor,
        "elevator": elevator,
        "district": district,                # real (if present)
        "district_guess": district_guess,    # heuristic (debug only)
        "url": url,
        "source": "otodom",
    }


# ----------------------------
# Detail page helpers
# ----------------------------

_PLN_INT_RE = re.compile(r"(\d[\d\s\u00A0]*)")  # digits + spaces + nbsp


def _to_int_pln(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x)
    m = _PLN_INT_RE.search(s)
    if not m:
        return None
    digits = m.group(1).replace(" ", "").replace("\u00A0", "")
    try:
        return int(digits)
    except ValueError:
        return None


def _extract_jsonld(soup: BeautifulSoup) -> list[dict]:
    """
    Extract JSON-LD blocks. Often contains address + geo.
    """
    out: list[dict] = []
    for tag in soup.select('script[type="application/ld+json"]'):
        txt = (tag.string or tag.get_text() or "").strip()
        if not txt:
            continue
        try:
            obj = json.loads(txt)
            if isinstance(obj, list):
                out.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _extract_geo_from_detail(
    next_data: Optional[dict],
    soup: BeautifulSoup,
) -> tuple[Optional[float], Optional[float]]:
    """
    GEO (lat/lon) is extracted ONLY from detail pages (Next.js or JSON-LD).
    """
    if next_data:
        props = next_data.get("props", {})
        nodes = _walk_find_dicts(props)
        for n in nodes:
            if not isinstance(n, dict):
                continue
            loc = n.get("location") or {}
            if not isinstance(loc, dict):
                continue
            coords = loc.get("coordinates") or loc.get("geo") or {}
            if not isinstance(coords, dict):
                continue

            cand_lat = coords.get("latitude") or coords.get("lat")
            cand_lon = coords.get("longitude") or coords.get("lon") or coords.get("lng")
            cand_lat = _to_number(cand_lat)
            cand_lon = _to_number(cand_lon)
            if cand_lat is not None and cand_lon is not None:
                return cand_lat, cand_lon

    for obj in _extract_jsonld(soup):
        geo = obj.get("geo") or {}
        if isinstance(geo, dict):
            cand_lat = _to_number(geo.get("latitude"))
            cand_lon = _to_number(geo.get("longitude"))
            if cand_lat is not None and cand_lon is not None:
                return cand_lat, cand_lon

        if obj.get("@type") == "Offer" and isinstance(obj.get("itemOffered"), dict):
            geo2 = obj["itemOffered"].get("geo") or {}
            if isinstance(geo2, dict):
                cand_lat = _to_number(geo2.get("latitude"))
                cand_lon = _to_number(geo2.get("longitude"))
                if cand_lat is not None and cand_lon is not None:
                    return cand_lat, cand_lon

    return None, None


def _find_best_detail_node(next_data: dict) -> Optional[dict]:
    """
    Otodom detail page is also Next.js. We scan dict nodes and pick the one
    that looks most like a full "ad" object.
    """
    props = next_data.get("props", {})
    nodes = _walk_find_dicts(props)

    best: Optional[dict] = None
    best_score = -1.0

    for n in nodes:
        if not isinstance(n, dict):
            continue
        if _pick(n, "title", "name") is None:
            continue

        score = 0.0
        if n.get("location"):
            score += 3
        if n.get("description") or n.get("body") or n.get("content"):
            score += 2
        if n.get("attributes") or n.get("characteristics"):
            score += 3
        if _pick(n, "price", "totalPrice", "priceFrom", "value") is not None:
            score += 2
        if n.get("images") or n.get("gallery"):
            score += 1

        attrs = n.get("attributes") or n.get("characteristics")
        if isinstance(attrs, list):
            score += min(len(attrs), 10) / 10.0

        if score > best_score:
            best_score = score
            best = n

    return best


def _format_street_pl(street: Optional[str]) -> Optional[str]:
    """
    Make street look like "ul. Myśliwska" when possible.
    If street already contains a prefix (ul., al., os., pl., etc.), keep it.
    """
    if not street or not isinstance(street, str):
        return None
    s = street.strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith(("ul.", "ul ", "aleja", "al.", "al ", "os.", "os ", "pl.", "pl ")):
        return s
    return f"ul. {s}"


def _extract_address_parts(address: Any) -> dict:
    """
    Normalize common address keys into a consistent set.
    Otodom keys can vary; we keep it tolerant.
    """
    if not isinstance(address, dict):
        return {}

    # Common variants observed across listings:
    # city, province, district, neighbourhood, quarter, county, municipality, postalCode, street, streetNumber
    parts = {
        "street": address.get("street"),
        "street_number": address.get("streetNumber") or address.get("houseNumber") or address.get("number"),
        "neighbourhood": address.get("neighbourhood") or address.get("neighborhood"),
        "quarter": address.get("quarter"),
        "district": address.get("district"),
        "city": address.get("city"),
        "municipality": address.get("municipality"),
        "county": address.get("county"),
        "province": address.get("province"),
        "postal_code": address.get("postalCode") or address.get("postcode") or address.get("zip"),
        "country": address.get("country"),
    }

    # Keep only strings
    for k, v in list(parts.items()):
        if v is None:
            continue
        if not isinstance(v, str):
            parts[k] = None
        else:
            parts[k] = v.strip() if v.strip() else None

    return parts


def _build_location_label(parts: dict) -> Optional[str]:
    """
    Build a human-friendly label like:
    "ul. Myśliwska, Płaszów, Podgórze, Kraków, małopolskie"
    """
    if not parts:
        return None

    street = _format_street_pl(parts.get("street"))
    # optionally add number
    if street and parts.get("street_number"):
        street = f"{street} {parts['street_number']}"

    items: list[str] = []
    if street:
        items.append(street)

    # order similar to UI: neighbourhood/quarter, district, city, province
    for key in ["neighbourhood", "quarter", "district", "city", "province"]:
        val = parts.get(key)
        if val and val not in items:
            items.append(val)

    label = ", ".join(items).strip()
    return label if label else None


def _extract_location_label_from_next_data(next_data: Optional[dict]) -> Optional[str]:
    """
    Some Next.js nodes may contain already formatted labels like:
    locationLabel / addressLabel / fullAddress etc.
    We scan for those.
    """
    if not next_data:
        return None

    props = next_data.get("props", {})
    nodes = _walk_find_dicts(props)

    candidates: list[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        for k in ["locationLabel", "addressLabel", "fullAddress", "formattedAddress", "addressLine"]:
            v = n.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

    # prefer longest (usually the most complete)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _parse_balcony_terrace_garden(x: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return (balcony, terrace, garden) as 1/0/None.
    If missing -> None.
    If string explicitly indicates none -> 0,0,0.
    If mentions specific items -> flags accordingly.
    """
    if x is None:
        return None, None, None
    s = str(x).strip().lower()
    if not s:
        return None, None, None

    if any(k in s for k in ["brak", "nie", "none", "no"]):
        return 0, 0, 0

    balcony = 1 if ("balkon" in s or "loggia" in s) else 0
    terrace = 1 if "taras" in s else 0
    garden = 1 if ("ogród" in s or "ogrod" in s or "garden" in s) else 0
    return balcony, terrace, garden


def _normalize_detail_fields(detail_node: dict, next_data: dict, soup: BeautifulSoup) -> dict:
    """
    Extract extra fields from listing detail.
    - Human-readable location_label is taken from detail page (Next.js/address),
      not guessed from title/URL.
    - GEO (lat/lon) is taken ONLY from detail page.
    """
    attrs = detail_node.get("attributes") or detail_node.get("characteristics")

    market = _get_attribute(attrs, ["rynek", "market"])
    ownership = _get_attribute(attrs, ["forma własności", "własność", "ownership"])
    build_year = _get_attribute(attrs, ["rok budowy", "year of construction", "year"])
    building_type = _get_attribute(attrs, ["rodzaj zabudowy", "building", "typ zabudowy"])
    material = _get_attribute(attrs, ["materiał", "material"])
    heating = _get_attribute(attrs, ["ogrzewanie", "heating"])
    finish_state = _get_attribute(attrs, ["stan wykończenia", "finish", "wykończenie"])
    windows = _get_attribute(attrs, ["okna", "windows"])
    parking = _get_attribute(attrs, ["miejsce parkingowe", "parking", "garaż", "garaz"])
    furnished = _get_attribute(attrs, ["umeblowanie", "furnished"])
    available_from = _get_attribute(attrs, ["dostępne od", "available from"])

    rent = _get_attribute(attrs, ["czynsz", "rent"])
    rent_pln = _to_int_pln(rent)

    balcony_terrace_garden_raw = _get_attribute(
        attrs,
        ["balkon", "taras", "ogród", "ogrod", "loggia"],
    )
    balcony, terrace, garden = _parse_balcony_terrace_garden(balcony_terrace_garden_raw)

    # location/address details (detail page)
    loc = detail_node.get("location") or {}
    address = (loc.get("address") or {}) if isinstance(loc, dict) else {}
    parts = _extract_address_parts(address)

    city = parts.get("city")
    street = parts.get("street")
    voivodeship = parts.get("province")
    neighbourhood = parts.get("neighbourhood")
    quarter = parts.get("quarter")
    district_detail = parts.get("district") or _extract_district(detail_node)
    postal_code = parts.get("postal_code")

    # "pretty" label like in UI (prefer existing label if available)
    location_label = _extract_location_label_from_next_data(next_data)
    if not location_label:
        location_label = _build_location_label(
            {
                "street": street,
                "street_number": parts.get("street_number"),
                "neighbourhood": neighbourhood,
                "quarter": quarter,
                "district": district_detail,
                "city": city,
                "province": voivodeship,
            }
        )

    # GEO (detail only)
    lat, lon = _extract_geo_from_detail(next_data, soup)

    # compute price per m2 if possible (uses values from detail node)
    price_obj = _pick(detail_node, "price", "totalPrice", "priceFrom", "value")
    price_pln, _currency = _normalize_price(price_obj)

    area = _pick(detail_node, "area", "areaInSquareMeters", "surface")
    area_m2 = _to_number(area)

    price_per_m2: Optional[int] = None
    if price_pln is not None and area_m2:
        try:
            price_per_m2 = int(round(price_pln / float(area_m2)))
        except ZeroDivisionError:
            price_per_m2 = None

    advertiser_type = _pick(detail_node, "advertiserType", "sellerType", "agency", "owner")

    desc = _pick(detail_node, "description", "body", "content")
    description_len = len(str(desc)) if desc is not None else None

    return {
        # precise location from detail
        "location_label": location_label,  # e.g. "ul. Myśliwska, Płaszów, Podgórze, Kraków, małopolskie"
        "street": street,
        "postal_code": postal_code,
        "neighbourhood": neighbourhood,
        "quarter": quarter,
        "city": city,
        "voivodeship": voivodeship,
        "district_detail": district_detail,

        # GEO only from detail
        "lat": lat,
        "lon": lon,

        # extended characteristics
        "market": market,
        "ownership": ownership,
        "build_year": int(_to_number(build_year)) if _to_number(build_year) is not None else None,
        "building_type": building_type,
        "material": material,
        "heating": heating,
        "finish_state": finish_state,
        "windows": windows,
        "parking": parking,
        "furnished": furnished,
        "available_from": available_from,

        # extras
        "rent_pln": rent_pln,
        "balcony": balcony,
        "terrace": terrace,
        "garden": garden,
        "price_per_m2": price_per_m2,
        "advertiser_type": advertiser_type,
        "description_len": description_len,
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


def enrich_with_details(
    df: pd.DataFrame,
    sleep_s: Optional[float] = None,
    max_listings: Optional[int] = None,
) -> pd.DataFrame:
    """
    Enrich an existing DataFrame (from collect_raw_listings) by visiting each detail page.
    Returns a new DataFrame with extra columns merged by URL.
    """
    if df is None or df.empty:
        return df

    if "url" not in df.columns:
        raise ValueError("DataFrame must contain 'url' column to enrich with details.")

    delay = CFG.detail_sleep_s if sleep_s is None else float(sleep_s)
    limit = max_listings if max_listings is not None else CFG.max_listings_details

    urls = df["url"].dropna().astype(str).tolist()
    if limit is not None:
        urls = urls[: int(limit)]

    detail_rows: list[dict] = []

    for url in urls:
        try:
            html = _fetch_html(url)
            soup = BeautifulSoup(html, "lxml")
            next_data = _extract_next_data(html)

            if not next_data:
                # If __NEXT_DATA__ is missing, we can still try JSON-LD GEO,
                # but all other "pretty location" fields may be missing.
                lat, lon = _extract_geo_from_detail(None, soup)
                detail_rows.append({"url": url, "lat": lat, "lon": lon})
            else:
                node = _find_best_detail_node(next_data)
                if not node:
                    lat, lon = _extract_geo_from_detail(next_data, soup)
                    detail_rows.append({"url": url, "lat": lat, "lon": lon})
                else:
                    extra = _normalize_detail_fields(node, next_data, soup)
                    extra["url"] = url
                    detail_rows.append(extra)

        except requests.HTTPError:
            detail_rows.append({"url": url})
        except Exception:
            detail_rows.append({"url": url})

        time.sleep(delay)

    details_df = pd.DataFrame(detail_rows).drop_duplicates(subset=["url"])
    out = df.merge(details_df, on="url", how="left")

    # final district: prefer detail, then structured list, then heuristic guess
    if "district_detail" in out.columns:
        out["district_final"] = out["district_detail"].fillna(out.get("district"))
        if "district_guess" in out.columns:
            out["district_final"] = out["district_final"].fillna(out["district_guess"])
    elif "district_guess" in out.columns:
        out["district_final"] = out.get("district").fillna(out["district_guess"])
    else:
        out["district_final"] = out.get("district")

    return out


def collect_listings_with_details(
    pages: int = 1,
    sleep_s: Optional[float] = None,
    detail_sleep_s: Optional[float] = None,
    max_details: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: list pages -> then enrich each listing with details.
    """
    df = collect_raw_listings(pages=pages, sleep_s=sleep_s)
    return enrich_with_details(df, sleep_s=detail_sleep_s, max_listings=max_details)
