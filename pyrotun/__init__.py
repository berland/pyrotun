import logging
import sys

logger = {}


def getLogger(name="pyrotun"):
    global logger
    if name == "__main__":
        name = "service"
    if len(name.split(".")) > 1:
        name = name.replace("pyrotun.", "")
    if "connections." in name:
        name = name.replace("connections.", "c.")
    if name not in logger:
        logger[name] = setup_logger(name)
        return logger[name]
    else:
        return logger[name]


def setup_logger(name="pyrotun"):
    shortname = name[:8]
    formatter = logging.Formatter(
        fmt="%(asctime)s " + f"{shortname:<8s}" + " %(levelname)-6s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler("log.txt", mode="w")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
