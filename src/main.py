import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from dataset.db_query import LeagueTag, SeasonTag
from dataset.dataset import SingleSeasonSingleLeague
from model.model import TeamEncoder, LSTMPredictionNet, DensePredictionNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set


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



def train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch_number):
    running_loss = 0.0
    running_loss_saved = list()
    steps = 20  # print running loss every k steps

    for i, (match, result) in enumerate(train_loader):
        optimizer.zero_grad()

        players_home = match["players_home"]
        players_away = match["players_away"]
    
        #goals_home = match["home_team_goal"]
        #goals_away = match["away_team_goal"]

        for x in players_home:
            x = x.to(device=device)
        for x in players_away:
            x = x.to(device=device)
        
        players_home_tensor = torch.stack(players_home, dim=1)
        players_home_tensor = players_home_tensor.view(players_home_tensor.shape[0], -1)
        players_home_tensor = players_home_tensor.to(device=device) 

        players_away_tensor = torch.stack(players_away, dim=1)
        players_away_tensor = players_away_tensor.view(players_away_tensor.shape[0], -1) 
        players_away_tensor = players_away_tensor.to(device=device) 

        pred_result = model(players_home_tensor, players_away_tensor)

        result = result.to(dtype=torch.float32, device=device)
    
        error = loss_fn(pred_result, result)
        error.backward()
        optimizer.step()

        running_loss += error.item()

        if i % steps == 0:
            running_loss = running_loss / steps
            running_loss_saved.append(running_loss)
            logging.info("epoch {} | step {} | running loss {}".format(epoch_number, i, running_loss))
            running_loss = 0.0


def validate(model, optimizer, loss_fn, device, valid_loader):
    losses = list()
    num_correct = 0
    with torch.no_grad():
        for i, (match, result) in enumerate(valid_loader):
            players_home = match["players_home"]
            players_away = match["players_away"]

            # send player vectors to device
            for x in players_home:
                x = torch.unsqueeze(x, 0)
                x = x.to(device=device)
            for x in players_away:
                x = torch.unsqueeze(x, 0)
                x = x.to(device=device)

            players_home_tensor = torch.stack(players_home, dim=1)
            players_home_tensor = players_home_tensor.view(players_home_tensor.shape[0], -1)
            players_home_tensor = players_home_tensor.to(device=device) 

            players_away_tensor = torch.stack(players_away, dim=1)
            players_away_tensor = players_away_tensor.view(players_away_tensor.shape[0], -1) 
            players_away_tensor = players_away_tensor.to(device=device) 

            pred_result = model(players_home_tensor, players_away_tensor)

            result = result.to(dtype=torch.float32, device=device)
            
            error = loss_fn(pred_result, result)
            losses.append(error.item())
            
            _, arg_max_idx = torch.max(pred_result, 1)
            if result[0, arg_max_idx] > 0:
                num_correct += 1

    return losses, num_correct


def train(model, optimizer, loss_fn, device, num_epochs, train_loader, valid_loader, batch_size):
    for epoch in range(num_epochs):
        model.train()
        train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch)

        model.eval()
        losses, num_correct = validate(model, optimizer, loss_fn, device, valid_loader)

        avg_loss = float(sum(losses)) / float(len(losses))
        acc = float(num_correct) / float(len(valid_loader)) 

        logging.info(
            "epoch {} | average validation loss {} | validation acc {}"
            .format(epoch, avg_loss, acc))

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
    if bool(args.big_dataset):
        train_set = make_train_set(sql_path)
        valid_set = make_valid_set(sql_path)
    else:
        train_set = make_small_train_set(sql_path)
        valid_set = make_small_valid_set(sql_path)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)

    # model = LSTMPredictionNet(
        # 35,
        # hidden_size=args.lstm_hidden_size)
    
    model = DensePredictionNet(11*35)

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
