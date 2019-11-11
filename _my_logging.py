import logging as _logger

logging = _logger.getLogger('my_layer')
logging.setLevel(_logger.INFO)
_hander = _logger.StreamHandler()
formatter = _logger.Formatter('[OL] %(message)s')
_hander.setFormatter(formatter)
logging.addHandler(_hander)


def info(fmt, *args):
    logging.info(fmt, *args)


def warning(fmt, *args):
    logging.warning(fmt, *args)


def debug(fmt, *args):
    logging.debug(fmt, *args)
