import time
from app.core.logger import logger


def send_welcome_email(email: str):
    time.sleep(2)
    logger.info("Welcome email sent to %s", email)
