import logging
import os

#Ensure logs directory exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

#Define log file path

LOG_FILE = os.path.join(LOG_DIR, "app.log")

#Configure loggging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format= "%(asctime)s - %(levelname)s - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("Sentiment Analysis")
