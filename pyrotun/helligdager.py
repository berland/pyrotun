import datetime
import holidays

import pyrotun.connections


def main():
    openhab = pyrotun.connections.openhab.OpenHABConnection()

    if datetime.datetime.now() in holidays.Norway():
        openhab.set_item("Fridag", "ON")
    else:
        openhab.set_item("Fridag", "OFF")
