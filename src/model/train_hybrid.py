import logging

import torch
from torch import nn

from model.model import DenseCompNet
from dataset.train_valid_test_loader import make_small_test_set, make_small_train_set, make_small_valid_set
from dataset.train_valid_test_loader import make_test_set, make_train_set, make_valid_set
from model.train_util import get_device, save_losses
from model.train_dcn import run_training_dcn
from model.train_dpn import run_training_dpn
from model.train_dpn import run_testing_dpn
import random


def run_training_hybrid(train_set_dcn, valid_set_dcn, train_set_dpn, valid_set_dpn, args, model=None):
    if model is None:
        model = run_training_dcn(train_set_dcn, valid_set_dcn, args)
    else:
        model = run_training_dcn(train_set_dcn, valid_set_dcn, args, model=model)

    # change last layer in this model to softmax prediction
    model.prediction = nn.Sequential(
        nn.Linear(model.hidden_size, model.hidden_size),
        nn.ReLU(),
        nn.Linear(model.hidden_size, 3),
        nn.Softmax(dim=1)
    )

    logging.info(str(model))
    
    model = run_training_dpn(train_set_dpn, valid_set_dpn, args, model=model)

    return model


def run_testing_hybrid(model, test_set, args):
    return run_testing_dpn(model, test_set, args)
