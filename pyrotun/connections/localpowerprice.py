"""Support functions for power prices that are local to the house"""

import datetime

import pandas as pd


def get_gridrental(priceindex: pd.DatetimeIndex, provider="bkk") -> pd.Series:
    """Data is valid forwards in time"""
    match provider:
        case "bkk":
            if 1 <= datetime.datetime.now().month <= 3:
                # winter
                night_rate = 0.3786
                day_rate = 0.5025
            else:
                # Summer
                night_rate = 0.5925
                day_rate = 0.4652
        case "tendranett":
            if 1 <= datetime.datetime.now().month <= 3:
                # winter
                night_rate = 0.306
                day_rate = 0.369
            else:
                # Summer
                night_rate = 0.393
                day_rate = 0.456
        case _:
            raise ValueError

    gridrental = pd.Series(index=priceindex, dtype=float)
    gridrental[:] = day_rate

    # Night rate:
    gridrental[priceindex.hour < 6] = night_rate
    gridrental[priceindex.hour >= 22] = night_rate

    # Weekend rate:
    gridrental[priceindex.weekday >= 5] = night_rate

    return gridrental
