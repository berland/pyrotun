import sys
import logging

# Singleton pattern for the logger
logger = None


def getLogger(name="pyrotun"):
    global logger
    if logger is None:
        logger = setup_logger(name)
        return logger
    else:
        return logger


def setup_logger(name="pyrotun"):
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler("log.txt", mode="w")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
