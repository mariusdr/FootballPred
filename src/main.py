import argparse
import logging

import torch

from dataset.db_query import LeagueTag, SeasonTag
from dataset.dataset import SingleSeasonSingleLeague
from model.model import TeamEncoder, LSTMPredictionNet, DensePredictionNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set
from model.train_dpn import run_training_dpn
from model.train_dcn import run_training_dcn

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
    
    logging.info(str(args))

    sql_path = args.database
    if bool(args.big_dataset):
        train_set = make_train_set(sql_path)
        valid_set = make_valid_set(sql_path)
    else:
        train_set = make_small_train_set(sql_path, undersample_probs=(0.65, 0.0, 1.0))
        valid_set = make_small_valid_set(sql_path, undersample_probs=(1.0, 0.0, 1.0))
    
    # run_training_dpn(train_set, valid_set, args)
    run_training_dcn(train_set, valid_set, args)

if __name__ == "__main__":
    main()
