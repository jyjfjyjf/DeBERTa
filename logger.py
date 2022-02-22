import logging
from config import log_path


def create_logger(cl_log_path):
    cl_logger = logging.getLogger(__name__)
    cl_logger.setLevel(logging.INFO)

    handler = logging.FileHandler(cl_log_path, encoding='utf-8')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    cl_logger.addHandler(handler)
    cl_logger.addHandler(console)

    return cl_logger


logger = create_logger(log_path)
