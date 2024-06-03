import logging
from paths import LOG_FILE_PATH

def setup_logging():
    logging.basicConfig(filename = LOG_FILE_PATH, 
                level = logging.DEBUG, 
                format='%(name)s :: %(asctime)s :: %(levelname)s :: %(message)s', 
                datefmt="%m/%d/%Y %I:%M:%S %p")