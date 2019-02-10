import argparse
import logging
import numpy as np
import torch
import pickle

from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set
from model.train_dpn import run_training_dpn, run_testing_dpn
from model.train_rnn_dpn import run_training_rnn_dpn, run_testing_rnn_dpn
from model.train_rnn import run_testing_rnn, run_training_rnn
from model.bookmaker_predictions import run_bookkeeper_tests
from model.train_dpn_with_odds import run_testing_dpn_odds, run_training_dpn_odds
from model.train_rnn_dpn_with_odds import run_testing_rnn_dpn_odds, run_training_rnn_dpn_odds

import random

parser = argparse.ArgumentParser("description = Football predictions using RNNs.")

# general config
parser.add_argument("--database", type=str, help="path to the training database")
parser.add_argument("--log", type=str, default=None, help="path to logfile ")
parser.add_argument("--loglevel", type=str, default="INFO", help="log level, either DEBUG or INFO")
parser.add_argument("--device", type=str, help="cuda or cpu", default="cpu")
parser.add_argument("--big_dataset", type=int, help="use the full dataset for this run", default=1)
parser.add_argument("--experiment", type=int, help="choose the experiment to run", default=1)


# training hyperparameters
parser.add_argument("--epochs", type=int, help="num. of training epochs")
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


def experiment1():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set.pkl")
        valid_set = deserialize("pickles/valid_set.pkl")
        test_set =  deserialize("pickles/test_set.pkl")
    else:
        train_set = deserialize("pickles/small_train_set_us.pkl")
        valid_set = deserialize("pickles/small_valid_set.pkl")
        test_set =  deserialize("pickles/small_test_set.pkl")

    model = run_training_dpn(train_set, valid_set, args)
    run_testing_dpn(model, test_set, args)
    logging.info(str(model))

def experiment2():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set_us.pkl")
        valid_set = deserialize("pickles/valid_set.pkl")
        test_set =  deserialize("pickles/test_set.pkl")
    else:
        train_set = deserialize("pickles/small_train_set_us.pkl")
        valid_set = deserialize("pickles/small_valid_set.pkl")
        test_set =  deserialize("pickles/small_test_set.pkl")

    model = run_training_dpn(train_set, valid_set, args)
    run_testing_dpn(model, test_set, args)
    logging.info(str(model))

def experiment3():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set_ts.pkl")
        valid_set = deserialize("pickles/valid_set_ts.pkl")
        test_set =  deserialize("pickles/test_set_ts.pkl")

    else:
        train_set = deserialize("pickles/small_train_set_ts.pkl")
        valid_set = deserialize("pickles/small_valid_set_ts.pkl")
        test_set =  deserialize("pickles/small_test_set_ts.pkl")

    model = run_training_rnn(train_set, valid_set, args)
    run_testing_rnn(model, test_set, args)
    logging.info(str(model))


def experiment4():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set_ts.pkl")
        valid_set = deserialize("pickles/valid_set_ts.pkl")
        test_set =  deserialize("pickles/test_set_ts.pkl")

    else:
        train_set = deserialize("pickles/small_train_set_ts.pkl")
        valid_set = deserialize("pickles/small_valid_set_ts.pkl")
        test_set =  deserialize("pickles/small_test_set_ts.pkl")

    model = run_training_rnn_dpn(train_set, valid_set, args)
    run_testing_rnn_dpn(model, test_set, args)
    logging.info(str(model))


def experiment5():
    if bool(args.big_dataset):
        test_set =  deserialize("pickles/test_set_ts_odds.pkl")

    else:
        test_set =  deserialize("pickles/small_test_set_ts_odds.pkl")
    
    run_bookkeeper_tests(test_set) 

def experiment6():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set_odds.pkl")
        valid_set = deserialize("pickles/valid_set_odds.pkl")
        test_set =  deserialize("pickles/test_set_odds.pkl")

    else:
        train_set = deserialize("pickles/small_train_set_odds.pkl")
        valid_set = deserialize("pickles/small_valid_set_odds.pkl")
        test_set =  deserialize("pickles/small_test_set_odds.pkl")

    model = run_training_dpn_odds(train_set, valid_set, args)
    run_testing_dpn_odds(model, test_set, args)
    logging.info(str(model))

def experiment7():
    if bool(args.big_dataset):
        train_set = deserialize("pickles/train_set_ts_odds.pkl")
        valid_set = deserialize("pickles/valid_set_ts_odds.pkl")
        test_set =  deserialize("pickles/test_set_ts_odds.pkl")

    else:
        train_set = deserialize("pickles/small_train_set_ts_odds.pkl")
        valid_set = deserialize("pickles/small_valid_set_ts_odds.pkl")
        test_set =  deserialize("pickles/small_test_set_ts_odds.pkl")
    
    model = run_training_rnn_dpn_odds(train_set, valid_set, args)
    run_testing_rnn_dpn_odds(model, test_set, args)
    logging.info(str(model))

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
    
    logging.info(args)

    if args.experiment == 1:
        experiment1()
    if args.experiment == 2:
        experiment2()
    if args.experiment == 3:
        experiment3()
    if args.experiment == 4:
        experiment4()
    if args.experiment == 5:
        experiment5()
    if args.experiment == 6:
        experiment6()
    if args.experiment == 7:
        experiment7()

def serialize(dataset, outfile):
    f = open(outfile, "wb")
    pickle.dump(dataset, f)
    f.close()

def deserialize(infile):
    f = open(infile, "rb")
    ds = pickle.load(f)
    f.close()
    return ds

def pickle_dump_datasets():
    tr = make_train_set(args.database, use_ts=True, with_odds=True)
    vl = make_valid_set(args.database, use_ts=True, with_odds=True)
    ts = make_test_set(args.database, use_ts=True, with_odds=True)
    
    serialize(tr, "train_set_ts_odds.pkl")
    serialize(vl, "valid_set_ts_odds.pkl")
    serialize(ts, "test_set_ts_odds.pkl")
    
    tr = make_small_train_set(args.database, use_ts=True, with_odds=True)
    vl = make_small_valid_set(args.database, use_ts=True, with_odds=True)
    ts = make_small_test_set(args.database, use_ts=True, with_odds=True)
    
    serialize(tr, "small_train_set_ts_odds.pkl")
    serialize(vl, "small_valid_set_ts_odds.pkl")
    serialize(ts, "small_test_set_ts_odds.pkl")

if __name__ == "__main__":
    main()
