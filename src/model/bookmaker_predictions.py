import numpy as np
import logging
from model.confusion_matrix import ConfusionMatrix
import torch

class BookkeeperPredictor(object):
    def __init__(self, id_string):
        self.id_string = id_string
        
        self.matches = 0
        self.incomplete = 0
        
        self.cfm = ConfusionMatrix()

    def predict(self, result, match):
        homeodds = match[self.id_string+"H"]
        drawodds = match[self.id_string+"D"]
        awayodds = match[self.id_string+"A"]
        
        if homeodds is None or drawodds is None or awayodds is None:
            self.incomplete += 1
            return
        self.matches += 1 
        
        odds = torch.tensor([homeodds, drawodds, awayodds])

        # convert odds to probabilities
        probs = 1.0 / odds
        
        _, predidx = torch.max(probs, dim=0)
        _, trueidx = torch.max(result, dim=0)
        
        self.cfm.insert(trueidx, predidx)

    def print_stats(self):
        logging.info("confusion matrix for bookkeeper {}".format(self.id_string))
        logging.info(self.cfm) 
        logging.info("absolute num. of predictions:")
        logging.info(self.cfm.print_data())
        logging.info("acc: {} ".format(self.cfm.get_acc()))
        logging.info("(bookkeeper has {} nan entries)".format(self.incomplete))

def run_bookkeeper_tests(test_set):
    bookkeepers = [
        BookkeeperPredictor("B365"),
        BookkeeperPredictor("BW"),
        BookkeeperPredictor("IW"),
        BookkeeperPredictor("LB"),
        BookkeeperPredictor("PS"),
        BookkeeperPredictor("WH"),
        BookkeeperPredictor("SJ"),
        BookkeeperPredictor("VC"),
        BookkeeperPredictor("GB")
    ]

    for i, (match, result) in enumerate(test_set):
        for bm in bookkeepers:
            bm.predict(result, match)

    for bm in bookkeepers:
        bm.print_stats()
