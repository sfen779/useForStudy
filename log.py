import logging
from logging.handlers import RotatingFileHandler
import os
import time
# from api.cardcheck_config import config as one_config


log_path = 'log'
if not os.path.exists(log_path):
   os.mkdir(log_path)
logger = logging.getLogger('fastapi')
logger.setLevel(logging.DEBUG)
format="%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s"
log_formatter = logging.Formatter(format)

file_name = 'server'+time.strftime('%Y-%m-%d',time.localtime(time.time())) + ".log"

info_handler = RotatingFileHandler(os.path.join(log_path,file_name),maxBytes=1*1024*1024,
                                                        backupCount=30)
info_handler.setFormatter(log_formatter)
    #info_handler.setLevel(one_config["log"]["level"])
info_handler.setLevel(logging.DEBUG)
logger.addHandler(info_handler)
