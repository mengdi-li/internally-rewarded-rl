import os
import sys
import datetime
import dateutil
import dateutil.tz

def prepare_dirs():
    dir_name = str(datetime.datetime.now(dateutil.tz.tzlocal())).split('+')[0]
    exp_dir = os.path.join('exps', dir_name)
    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    return exp_dir


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        pass    

