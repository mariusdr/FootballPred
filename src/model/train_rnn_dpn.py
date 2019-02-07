import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from model.model import RecurrentPredictionNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set
from model.train_util import get_device, save_losses

import random

def train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch_number):
    # running_loss = 0.0
    # steps = 5  # print running loss every k steps
    saved_losses = list()
    for i, (match, result) in enumerate(train_loader):
        optimizer.zero_grad()

        players_home = match["players_home"]
        players_away = match["players_away"]

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
    
        
        history_home = match["home_team_history"]
        history_away = match["away_team_history"]
        for i in range(len(history_home)):
            history_home[i] = history_home[i].to(device=device)
        for i in range(len(history_away)):
            history_away[i] = history_away[i].to(device=device)


        hidden1 = model.hist_enc._init_hidden(players_home_tensor.shape[0], device)
        hidden2 = model.hist_enc._init_hidden(players_away_tensor.shape[0], device)
        pred_result = model(players_home_tensor, players_away_tensor, history_home, history_away, hidden1, hidden2)
        
        result = result.to(dtype=torch.float32, device=device)

        error = loss_fn(pred_result, result)
        error.backward()
        optimizer.step()

        saved_losses.append(error.item())

        # running_loss += error.item()
        # if i % steps == 0 and i > 0:
        # running_loss = running_loss / steps
        # logging.info("epoch {} | step {} | running loss {}".format(epoch_number, i, running_loss))
        # running_loss = 0.0

    return saved_losses

def validate(model, optimizer, loss_fn, device, valid_loader):
    losses = list()
    num_correct = 0

    # count how often results were correctly predicted
    predicted_home_win = 0
    predicted_away_win = 0
    predicted_draw = 0

    total_home_win = 0
    total_away_win = 0
    total_draw = 0

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

            history_home = match["home_team_history"]
            history_away = match["away_team_history"]
            for i in range(len(history_home)):
                history_home[i] = history_home[i].to(device=device)
            for i in range(len(history_away)):
                history_away[i] = history_away[i].to(device=device)
            
            hidden1 = model.hist_enc._init_hidden(players_home_tensor.shape[0], device=device)
            hidden2 = model.hist_enc._init_hidden(players_away_tensor.shape[0], device=device)
            pred_result = model(players_home_tensor, players_away_tensor, history_home, history_away, hidden1, hidden2)

            result = result.to(dtype=torch.float32, device=device)

            error = loss_fn(pred_result, result)
            losses.append(error.item())

            _, arg_max_idx = torch.max(pred_result, 1)
            if result[0, arg_max_idx] > 0:
                num_correct += 1

                if arg_max_idx == 0:
                    predicted_home_win += 1
                if arg_max_idx == 1:
                    predicted_draw += 1
                if arg_max_idx == 2:
                    predicted_away_win += 1

            if result[0, 0] > 0:
                total_home_win += 1
            if result[0, 1] > 0:
                total_draw += 1
            if result[0, 2] > 0:
                total_away_win += 1

        logging.info(
            "predicted (home, draw, away) win correctly: {}/{} | {}/{} | {}/{}"
            .format(predicted_home_win, total_home_win, predicted_draw,
                    total_draw, predicted_away_win, total_away_win))
    return losses, num_correct


def train(model, optimizer, loss_fn, device, num_epochs, train_loader, valid_loader, batch_size, model_save_path, stats_save_path):
    losses = list()
    for epoch in range(num_epochs):
        model.train()
        train_losses = train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch)
        avg_train_loss = float(sum(train_losses)) / float(len(train_losses))

        model.eval()
        valid_losses, valid_num_correct = validate(model, optimizer, loss_fn, device, valid_loader)
        avg_valid_loss = float(sum(valid_losses)) / float(len(valid_losses))
        valid_acc = float(valid_num_correct) / float(len(valid_loader))
        logging.info(
            "epoch {} | average train loss {} | average validation loss {} | validation acc {}"
            .format(epoch, avg_train_loss, avg_valid_loss, valid_acc))


        if model_save_path is not None:
            savestate = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }
            savepath = model_save_path + "/dpn_checkpoint_epoch{}.pth".format(epoch)
            torch.save(savestate, savepath)

        losses.append((avg_train_loss, avg_valid_loss, valid_acc))

    if stats_save_path is not None:
        save_losses(losses, stats_save_path + "/dpn_train_stats.txt")


def run_training_rnn_dpn(train_set, valid_set, args, model=None):
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)
    
    if model is None:
        model = RecurrentPredictionNet(11*35, 2)
    
    model.hist_enc.to(device)
    model.dpn.to(device)
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
          valid_loader, args.batch_size, args.model_save_path,
          args.stats_save_path)

    return model

def test(model, device, test_loader):
    num_correct = 0

    # count how often results were correctly predicted
    predicted_home_win = 0
    predicted_away_win = 0
    predicted_draw = 0

    total_home_win = 0
    total_away_win = 0
    total_draw = 0

    with torch.no_grad():
        for i, (match, result) in enumerate(test_loader):
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

            history_home = match["home_team_history"]
            history_away = match["away_team_history"]
            for i in range(len(history_home)):
                history_home[i] = history_home[i].to(device=device)
            for i in range(len(history_away)):
                history_away[i] = history_away[i].to(device=device)

            hidden1 = model.hist_enc._init_hidden(players_home_tensor.shape[0], device=device)
            hidden2 = model.hist_enc._init_hidden(players_away_tensor.shape[0], device=device)
            pred_result = model(players_home_tensor, players_away_tensor, history_home, history_away, hidden1, hidden2)

            result = result.to(dtype=torch.float32, device=device)

            _, arg_max_idx = torch.max(pred_result, 1)
            if result[0, arg_max_idx] > 0:
                num_correct += 1

                if arg_max_idx == 0:
                    predicted_home_win += 1
                if arg_max_idx == 1:
                    predicted_draw += 1
                if arg_max_idx == 2:
                    predicted_away_win += 1

            if result[0, 0] > 0:
                total_home_win += 1
            if result[0, 1] > 0:
                total_draw += 1
            if result[0, 2] > 0:
                total_away_win += 1

        logging.info(
            "predicted (home, draw, away) win correctly: {}/{} | {}/{} | {}/{}"
            .format(predicted_home_win, total_home_win, predicted_draw,
                    total_draw, predicted_away_win, total_away_win))
    return num_correct


def run_testing_rnn_dpn(model, test_set, args):
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)

    num_correct = test(model, device, test_loader)
    acc = float(num_correct) / float(len(test_loader))
    logging.info("testing accuracy: {}".format(acc))


