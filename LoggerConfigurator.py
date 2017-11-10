
import logging
from Enumerators import DebugLevel

def configure_log(app_name, debug=DebugLevel.VERBOSE):
    logger = logging.getLogger(app_name)
    ch = logging.StreamHandler()

    if debug == DebugLevel.VERBOSE:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif debug==DebugLevel.LIGHT:
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)

    # create console handler and set level to debug


    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger