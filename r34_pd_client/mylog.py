#coding:utf-8

import os,sys
import re
import logging
from logging.handlers import TimedRotatingFileHandler

class MyLog(object):
    def __init__(self, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        
        log_filename = '/log/medical_Service.log'
        fileTimeHandler = TimedRotatingFileHandler(log_filename,"midnight",1,7)
        fileTimeHandler.suffix = "%Y-%m-%d.log"
        fileTimeHandler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        fileTimeHandler.setFormatter(formatter)
        self.logger.addHandler(fileTimeHandler)
        fileTimeHandler.close()

    def getlog(self):
        return self.logger
