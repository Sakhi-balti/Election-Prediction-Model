

import logging
import os
from datetime import datetime


LOG_FILE = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+ '.log'

log_folder = os.path.join(os.getcwd(), 'log')
os.makedirs(log_folder, exist_ok=True)

log_file_path = os.path.join(log_folder, LOG_FILE)

logging.basicConfig(
    filename = log_file_path,
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s -%(message)s"

)

logging.info("This is a test log message!")
