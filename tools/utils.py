import logging

def configure_logging(name: str = None, verbose: bool = False):
    """Setup logging."""
    logging_level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger(__name__ if name is None else name)
    logging.basicConfig(level=logging_level)
    root_logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    return root_logger