import argparse
import logging
import numpy as np
import torch

from dataset.db_query import LeagueTag, SeasonTag
from dataset.dataset import SingleSeasonSingleLeague
from model.model import TeamEncoder, LSTMPredictionNet, DensePredictionNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set
from model.train_dpn import run_training_dpn, run_testing_dpn
from model.train_dcn import run_training_dcn, run_testing_dcn
from model.train_hybrid import run_training_hybrid, run_testing_hybrid
from model.train_rnn_dpn import run_training_rnn_dpn, run_testing_rnn_dpn

import random

parser = argparse.ArgumentParser("description = Football predictions using RNNs.")

# general config
parser.add_argument("--database", type=str, help="path to the training database")
parser.add_argument("--log", type=str, default=None, help="path to logfile ")
parser.add_argument("--loglevel", type=str, default="INFO", help="log level, either DEBUG or INFO")
parser.add_argument("--device", type=str, help="cuda or cpu", default="cpu")
parser.add_argument("--big_dataset", type=int, help="use the full dataset for this run", default=1)

# model hyperparameters
parser.add_argument("--lstm_hidden_size", type=int, help="num. of neurons in the hidden layers of the LSTM encoder", default=128)
parser.add_argument("--lstm_hidden_layers", type=int, help="num. of hidden layers in the LSTM encoder", default=1)
parser.add_argument("--bidirectional", type=bool, help="use bidirectional LSTM", default=False)

# training hyperparameters
parser.add_argument("--epochs", type=int, help="num. of training epochs")
parser.add_argument("--dropout_p", type=float, help="dropout probability", default=0.5)
parser.add_argument("--batch_size", type=int, help="batch size for training", default=1)
parser.add_argument("--shuffle", type=bool, help="shuffle data before training", default=False)
parser.add_argument("--optimizer", type=str, help="either Adam or SGD", default="SGD")
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.001)
parser.add_argument("--momentum", type=float, help="momentum, only if optimizer is SGD", default=0.0)
parser.add_argument("--nesterov", type=bool, help="use nesterov accelerated gradient, only if optimizer is SGD", default=False)
parser.add_argument("--weight_decay", type=float, help="L2 regularization for Adam optimizer", default=0.0)

# saving models and training statistics
parser.add_argument("--model_save_path", type=str, help="path to saved model weights, if None models won't be saved", default=None)
parser.add_argument("--stats_save_path", type=str, help="path to saved training statistics, if None they won't be saved", default=None)
args = parser.parse_args()



def main():
    handlers = [logging.StreamHandler()]
    if args.log is not None:
        handlers.append(logging.FileHandler(args.log, mode="w"))
    loglevel = logging.INFO
    if args.loglevel == "DEBUG":
        loglevel = logging.DEBUG
    logging.basicConfig(
        handlers=handlers,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=loglevel)

    sql_path = args.database
    train_us_probs = (1.0, 1.0, 1.0)
    test_us_probs = (1.0, 1.0, 1.0)

    use_ts=False

    logging.info(
        "undersampling with probabilities: train = {} | test = {}".format(
            train_us_probs, test_us_probs))
    logging.info("using timeslices = {}".format(use_ts))

    # ts = make_test_set(sql_path, undersample_probs=test_us_probs, use_ts=False, with_odds=True)
    # baseline(ts)
    # return


    if bool(args.big_dataset):
        train_set = make_train_set(sql_path, undersample_probs=train_us_probs, use_ts=use_ts)
        valid_set = make_valid_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)
        test_set = make_test_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)

        # train_set_no_draws = make_train_set(sql_path, undersample_probs=(0.6, 0.0, 1.0))
        # valid_set_no_draws = make_valid_set(sql_path, undersample_probs=(0.6, 0.0, 1.0))
    else:
        #train_set = make_small_train_set(sql_path, undersample_probs=train_us_probs, use_ts=use_ts)
        #valid_set = make_small_valid_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)
        #test_set = make_small_test_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)
        pass
    
    train_set = make_train_set(sql_path, undersample_probs=train_us_probs, use_ts=use_ts)
    valid_set = make_valid_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)
    test_set = make_test_set(sql_path, undersample_probs=test_us_probs, use_ts=use_ts)
    
    serialize(train_set, "train_set.pkl")
    serialize(valid_set, "valid_set.pkl")
    serialize(test_set, "test_set.pkl")
    

    def run_experiment():
        lrs = [0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.075, 0.1]
        for lr in lrs:
            args.learning_rate = lr
            logging.info(str(args))
            model = run_training_dpn(train_set, valid_set, args)
            run_testing_dpn(model, test_set, args)
    
    #run_experiment()    

    #model = run_training_dcn(train_set, valid_set, args)
    #run_testing_dcn(model, test_set, args)

    # model = run_training_hybrid(train_set_no_draws, valid_set_no_draws, train_set, valid_set, args)
    # run_testing_hybrid(model, test_set, args)

    #model = run_training_rnn_dpn(train_set, valid_set, args)
    #run_testing_rnn_dpn(model, test_set, args)

    #logging.info(str(model))


        # X = {
            # "home_team_name": match["home_team_long_name"],
            # "away_team_name": match["away_team_long_name"],
            # "team_home": encoded_team_home,
            # "team_away": encoded_team_away,
            # "players_home": encoded_players_home,
            # "players_away": encoded_players_away,
            # "home_team_goal": match["home_team_goal"],
            # "away_team_goal": match["away_team_goal"]
        # }


import pickle

def serialize(dataset, outfile):
    f = open(outfile, "wb")
    pickle.dump(dataset, f)
    f.close()

def deserialize(infile):
    f = open(infile, "rb")
    ds = pickle.load(f)
    f.close()
    return ds

class Bookkeeper(object):
    def __init__(self, id_string):
        self.id_string = id_string

        self.incorrect = 0

        self.hw_correct = 0
        self.aw_correct = 0
        self.dw_correct = 0

        self.num_hws = 0
        self.num_aws = 0
        self.num_dws = 0

        self.incomplete = 0

    def _classify(self, match):
        hstr = self.id_string + "H"
        dstr = self.id_string + "D"
        astr = self.id_string + "A"

        ho = match[hstr]
        do = match[dstr]
        ao = match[astr]

        if ho is None or do is None or ao is None:
            return -1

        if np.isnan(ho) or np.isnan(do) or np.isnan(ao):
            return -1
    
        # convert odds to probabilities
        hp = 1.0 / ho
        dp = 1.0 / do
        ap = 1.0 / ao

        logging.info("{} odds: H {} D {} A {}".format(self.id_string, ho, do, ao))
        logging.info("{} probs: H {} D {} A {}".format(self.id_string, hp, dp, ap))
        
        if hp >= dp and hp >= ap:
            return 0
        if dp >= hp and dp >= ap:
            return 1
        if ap >= hp and ap >= dp:
            return 2

    def _cnt(self, result):
        if result[0] > 0:
            self.num_hws += 1
        if result[1] > 0:
            self.num_dws += 1
        if result[2] > 0:
            self.num_aws += 1

    def predict(self, result, match):
        pidx = self._classify(match)
        
        logging.info("{} | {}".format(self.id_string, result))

        if pidx < 0:
            self.incomplete += 1
            return

        if result[pidx] > 0:
            if pidx == 0:
                self.hw_correct += 1
            if pidx == 1:
                self.dw_correct += 1
            if pidx == 2:
                self.aw_correct += 1
        else:
            self.incorrect += 1

        self._cnt(result)

    def get_acc(self):
        n = self.num_aws + self.num_dws + self.num_hws
        c = self.hw_correct + self.dw_correct + self.aw_correct
        
        if n > 0:
            acc = float(c) / float(n)
        else:
            acc = -1.0
        logging.info("accuracy of {}: {}".format(self.id_string, acc))

        logging.info(
            "results of {}: H {}/{}, D {}/{}, A {}/{}   [incomplete {}]".format(
                self.id_string, self.hw_correct, self.num_hws, self.dw_correct,
                self.num_dws, self.aw_correct, self.num_aws, self.incomplete))

        return acc

def baseline(test_set):
    bookkeepers = [
        Bookkeeper("B365"),
        Bookkeeper("BW"),
        Bookkeeper("IW"),
        Bookkeeper("LB"),
        Bookkeeper("PS"),
        Bookkeeper("WH"),
        Bookkeeper("WH"),
        Bookkeeper("SJ"),
        Bookkeeper("VC"),
        Bookkeeper("GB")
    ]

    for i, (match, result) in enumerate(test_set):
        for bm in bookkeepers:
            bm.predict(result, match)

    for bm in bookkeepers:
        bm.get_acc()

if __name__ == "__main__":
    main()
