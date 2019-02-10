import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from model.model import DensePredictionNet
from model.train_util import get_device, save_losses
from model.confusion_matrix import ConfusionMatrix


def train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch_number):
    saved_losses = list()
    for i, (match, result) in enumerate(train_loader):
        optimizer.zero_grad()

        players_home = match["players_home"]
        players_away = match["players_away"]

        players_home_tensor = torch.stack(players_home, dim=1) 
        players_away_tensor = torch.stack(players_away, dim=1)
        
        players_home_tensor = players_home_tensor.view(players_home_tensor.shape[0], -1)
        players_away_tensor = players_away_tensor.view(players_away_tensor.shape[0], -1)

        players_home_tensor = players_home_tensor.to(device=device)
        players_away_tensor = players_away_tensor.to(device=device)

        pred_result = model(players_home_tensor, players_away_tensor)
        result = result.to(dtype=torch.float32, device=device)

        error = loss_fn(pred_result, result)
        error.backward()
        optimizer.step()

        saved_losses.append(error.item())

    return saved_losses


def validate(model, loss_fn, device, valid_loader, testing=False):
    losses = list()
    cfm = ConfusionMatrix()

    with torch.no_grad():
        for i, (match, result) in enumerate(valid_loader):
            players_home = match["players_home"]
            players_away = match["players_away"]

            players_home_tensor = torch.stack(players_home, dim=1) 
            players_away_tensor = torch.stack(players_away, dim=1) 

            players_home_tensor = players_home_tensor.to(device=device)
            players_away_tensor = players_away_tensor.to(device=device)
            
            players_home_tensor = players_home_tensor.view(players_home_tensor.shape[0], -1)
            players_away_tensor = players_away_tensor.view(players_away_tensor.shape[0], -1)
            
            pred_result = model(players_home_tensor, players_away_tensor)
            result = result.to(dtype=torch.float32, device=device)
            
            if not testing:
                error = loss_fn(pred_result, result)
                losses.append(error.item())
            
            _, ridx = torch.max(result, 1)
            _, pidx = torch.max(pred_result, 1)
            
            cfm.insert(ridx, pidx)

    logging.info("class specific acc for H {:.2f} D {:.2f} A {:.2f}".format(cfm.class_acc(0), cfm.class_acc(1), cfm.class_acc(2)))
    return losses, cfm


def train(model, optimizer, loss_fn, device, num_epochs, train_loader, valid_loader, batch_size, model_save_path, stats_save_path):
    losses = list()
    for epoch in range(num_epochs):
        model.train()
        train_losses = train_one_epoch(model, optimizer, loss_fn, device, train_loader, valid_loader, batch_size, epoch)
        avg_train_loss = float(sum(train_losses)) / float(len(train_losses))

        model.eval()
        valid_losses, valid_cfm = validate(model, loss_fn, device, valid_loader)
        avg_valid_loss = float(sum(valid_losses)) / float(len(valid_losses))
        valid_acc = valid_cfm.get_acc()
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


def run_training_dpn(train_set, valid_set, args, model = None):
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)
    
    if model is None:
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
          valid_loader, args.batch_size, args.model_save_path,
          args.stats_save_path)

    return model

def run_testing_dpn(model, test_set, args):
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.device == "cuda":
        device = get_device(use_cuda=True)
    else:
        device = get_device(use_cuda=False)

    _, cfm = validate(model, None, device, test_loader, testing=True)
    logging.info("testing accuracy: {}".format(cfm.get_acc()))
    logging.info("testing confusion matrix: \n{}".format(cfm))
    


