import torch
import logging

def get_device(use_cuda):
    if use_cuda and torch.cuda.is_available():
        default_cuda = torch.device("cuda")
        logging.info("cuda device is available, selected {}".format(default_cuda))
        return default_cuda
    else:
        default_cpu = torch.device("cpu")
        logging.info("cuda device is unavailable, selected {}".format(default_cpu))
        return default_cpu

def save_losses(losses, path):
    with open("path", "w") as f:
        for i in losses:
            f.write("{}\n".format(i))
