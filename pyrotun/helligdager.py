import datetime
import holidays

import pyrotun
import pyrotun.connections.openhab

logger = pyrotun.getLogger(__name__)


def main(connections=None):
    if connections is None:
        connections = {}
        connections["openhab"] = pyrotun.connections.openhab.OpenHABConnection()

    if datetime.datetime.now() in holidays.Norway():
        logger.info("Det er fridag")
        connections["openhab"].set_item("Fridag", "ON")
    else:
        logger.info("Det er ikke fridag")
        connections["openhab"].set_item("Fridag", "OFF")


if __name__ == "__main__":
    main()
