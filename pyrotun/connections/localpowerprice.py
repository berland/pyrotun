"""Support functions for power prices that are local to the house"""

import pandas as pd


def get_gridrental(priceindex: pd.DatetimeIndex) -> pd.Series:
    """Data is valid forwards in time"""
    # BKK:
    high_rate = 0.499
    low_rate = 0.399

    gridrental = pd.Series(index=priceindex, dtype=float)
    gridrental[:] = high_rate

    # Night rate:
    gridrental[priceindex.hour < 6] = low_rate
    gridrental[priceindex.hour >= 22] = low_rate

    # Weekend rate:
    gridrental[priceindex.weekday >= 5] = low_rate

    return gridrental
