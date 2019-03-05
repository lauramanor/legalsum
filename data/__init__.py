import os.path
from os import walk
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import csv


class Summaries():

    #tosdr_annotated code guide
    annotations = ["1", "2", "3", "s", "j", "d", "o", "q"]

    # handeled = {'tosdr_annotated':True, 'tldrlegal':True}

    def __init__(self, loads=[]):
        self.items = {}
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.files = {}

        for load in loads:
            self.load(load)

    def load(self, file):
        self.check_dir()
        reader = csv.DictReader(open(self.files[file]))
        for row in reader:
            row['ref'] = row['summary']
            self.items[row['uid']] = row



            pass

    def check_dir(self, check=None):
        if not check:
            check = self.dir_path

        for (dirpath, dirnames, filenames) in walk(check):
            for file in filenames:
                if file.endswith(".csv"):
                    self.files[file[:-4]] = os.path.join(dirpath, file)
        logging.debug(f'Checking {check}')
        logging.debug(', '.join(self.files))



if __name__ == '__main__':
    logging.debug(f'Invoking __init__.py for {__name__}')

