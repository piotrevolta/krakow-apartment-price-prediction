"""
Scraping module for collecting apartment listing data.

For now this module returns a small dummy dataset to validate the pipeline.
Later it will be replaced with real scraping logic.
"""

from datetime import datetime
import pandas as pd


def collect_raw_listings() -> pd.DataFrame:
    """
    Collect raw apartment listing data.

    Returns
    -------
    pandas.DataFrame
        Raw dataset with apartment listings (dummy for now).
    """
    now = datetime.utcnow().isoformat()

    data = [
        {
            "source": "dummy",
            "city": "Kraków",
            "district": "Stare Miasto",
            "price_pln": 950000,
            "area_m2": 52.0,
            "rooms": 2,
            "floor": 3,
            "created_at_utc": now,
        },
        {
            "source": "dummy",
            "city": "Kraków",
            "district": "Podgórze",
            "price_pln": 720000,
            "area_m2": 45.0,
            "rooms": 2,
            "floor": 1,
            "created_at_utc": now,
        },
        {
            "source": "dummy",
            "city": "Kraków",
            "district": "Krowodrza",
            "price_pln": 1100000,
            "area_m2": 68.5,
            "rooms": 3,
            "floor": 5,
            "created_at_utc": now,
        },
    ]

    return pd.DataFrame(data)
