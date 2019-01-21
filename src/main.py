import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from dataset.db_query import LeagueTag, SeasonTag
from dataset.dataset import SingleSeasonSingleLeague
from model.train import train, get_device
from model.model import TeamEncoder, SiamesePredictionNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set


parser = argparse.ArgumentParser("description = Football predictions using RNNs.")

# general config
parser.add_argument("--database", type=str, help="path to the training database")
parser.add_argument("--log", type=str, default=None, help="path to logfile ")
parser.add_argument("--loglevel", type=str, default="INFO", help="log level, either DEBUG or INFO")
parser.add_argument("--device", type=str, help="cuda or cpu", default="cpu")
parser.add_argument("--big_dataset", type=bool, help="use the full dataset for this run", default=True)

# model hyperparameters
parser.add_argument("--lstm_hidden_size", type=int, help="num. of neurons in the hidden layers of the LSTM encoder", default=128)
parser.add_argument("--lstm_hidden_layers", type=int, help="num. of hidden layers in the LSTM encoder", default=1)
parser.add_arugment("--bidirectional", type=bool, help="use bidirectional LSTM", default=False)

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

def train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch_number):
    running_loss = 0.0

    for i, (match, result) in enumerate(train_loader):
        optimizer.zero_grad()

        players_home = match_dict["players_home"]
        players_away = match_dict["players_away"]

        pred_result = model(players_home, players_away)
        
        result = result.to(dtype=torch.float32)
        if batch_size == 1:
            result = torch.unsqueeze(result, 0)

        error = loss_fn(pred_result,
                        torch.unsqueeze(result.to(dtype=torch.float32), 0))
        error.backward()

        optimizer.step()

        running_loss += error.item()

        if i > 0 and i % 150 == 0:
            print("epoch {} | step {} | running loss {}".format(epoch_number, i, running_loss / 150))
            running_loss = 0.0

def train(model, optimizer, loss_fn, device, num_epochs, train_loader, valid_loader, batch_size):
    for epoch in range(num_epochs):
        model.train()
        train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch)

def get_device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        default_cuda = torch.device("cuda")
        logging.info("cuda device is available, selected {}".format(default_cuda))
        return default_cuda
    else:
        default_cpu = torch.device("cpu")
        logging.info("cuda device is unavailable, selected {}".format(default_cpu))
        return default_cpu

def run_training():
    sql_path = args.database
    if args.big_dataset:
        train_set = make_train_set(sql_path)
        valid_set = make_valid_set(sql_path)
    else:
        train_set = make_small_train_set(sql_path)
        valid_set = make_small_valid_set(sql_path)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=args.shuffle)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)

    model = SiamesePredictionNet(35, args.hidden_size, args.num_hidden_layers)
    model.to(device)

    if args.optimizer == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
    else:
        optimizer = SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            nesterov=args.nesterov)

    loss_fn = torch.nn.BCELoss()
    train(model, optimizer, loss_fn, device, args.epochs, train_loader,
          valid_loader, args.batch_size)


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

    run_training()

if __name__ == "__main__":
    main()
