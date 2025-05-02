import os
import logging

LOG_PATH = "logs/app.log"

def get_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(LOG_PATH)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
