import re
from src.utils.logger import logger
def clean_text(text):
    logger.info(f"Cleaning text: {text[:50]}..")

    print('Remove URLs')
    text = re.sub(r'http\S+|www\S+|https\S+','',text, flags = re.MULTILINE)
    print('Remove special characters (keep revelant ones like $, %)')
    text = re.sub(r'[^\w\s$%]','',text)
    print('Convert to lowercase')
    text = text.lower()
    
    logger.info(f"Cleaned text: {text[:50]}..")
    return text