import os
import sys
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils import data
from torch import Tensor

from datetime import datetime
import itertools
import sqlite3
import logging
import time

from dataset.db_query import LeagueTag, SeasonTag
from dataset.dataset import SingleSeasonSingleLeague

"""
All together we have 75 seasons in different leagues (minus the ones 
we don't use because they have to many defect entries)

Each Seasons has around 316 matches (~ Bundesliga) probably all together 
around 24000 matches ...

We do a 80/20 Train-Test Split where we use some of the Train seasons 
for Validation.

i.e. 
    50 for Train
    10 for Valid
    15 for Test
"""

TRAIN_SEASONS = [(LeagueTag.NET, SeasonTag.S14_15),
                 (LeagueTag.SPA, SeasonTag.S08_09),
                 (LeagueTag.NET, SeasonTag.S13_14),
                 (LeagueTag.SCO, SeasonTag.S12_13),
                 (LeagueTag.SCO, SeasonTag.S15_16),
                 (LeagueTag.NET, SeasonTag.S10_11),
                 (LeagueTag.BEL, SeasonTag.S15_16),
                 (LeagueTag.ITA, SeasonTag.S08_09),
                 (LeagueTag.ENG, SeasonTag.S09_10),
                 (LeagueTag.BEL, SeasonTag.S09_10),
                 (LeagueTag.NET, SeasonTag.S09_10),
                 (LeagueTag.ITA, SeasonTag.S12_13),
                 (LeagueTag.FRA, SeasonTag.S14_15),
                 (LeagueTag.BEL, SeasonTag.S14_15),
                 (LeagueTag.ITA, SeasonTag.S14_15),
                 (LeagueTag.ITA, SeasonTag.S15_16),
                 (LeagueTag.GER, SeasonTag.S13_14),
                 (LeagueTag.ITA, SeasonTag.S09_10),
                 (LeagueTag.SWI, SeasonTag.S11_12),
                 (LeagueTag.FRA, SeasonTag.S12_13),
                 (LeagueTag.ENG, SeasonTag.S13_14),
                 (LeagueTag.FRA, SeasonTag.S15_16),
                 (LeagueTag.SPA, SeasonTag.S10_11),
                 (LeagueTag.SWI, SeasonTag.S15_16),
                 (LeagueTag.SPA, SeasonTag.S13_14),
                 (LeagueTag.SCO, SeasonTag.S08_09),
                 (LeagueTag.SPA, SeasonTag.S09_10),
                 (LeagueTag.NET, SeasonTag.S11_12),
                 (LeagueTag.SPA, SeasonTag.S14_15),
                 (LeagueTag.ENG, SeasonTag.S14_15),
                 (LeagueTag.SCO, SeasonTag.S10_11),
                 (LeagueTag.ITA, SeasonTag.S11_12),
                 (LeagueTag.SCO, SeasonTag.S14_15),
                 (LeagueTag.POR, SeasonTag.S10_11),
                 (LeagueTag.NET, SeasonTag.S15_16),
                 (LeagueTag.SPA, SeasonTag.S11_12),
                 (LeagueTag.FRA, SeasonTag.S13_14),
                 (LeagueTag.GER, SeasonTag.S14_15),
                 (LeagueTag.GER, SeasonTag.S12_13),
                 (LeagueTag.BEL, SeasonTag.S13_14),
                 (LeagueTag.SCO, SeasonTag.S11_12),
                 (LeagueTag.POR, SeasonTag.S15_16),
                 (LeagueTag.POR, SeasonTag.S12_13),
                 (LeagueTag.GER, SeasonTag.S09_10),
                 (LeagueTag.BEL, SeasonTag.S10_11),
                 (LeagueTag.SWI, SeasonTag.S09_10),
                 (LeagueTag.ITA, SeasonTag.S10_11),
                 (LeagueTag.ENG, SeasonTag.S15_16),
                 (LeagueTag.SWI, SeasonTag.S12_13),
                 (LeagueTag.BEL, SeasonTag.S11_12),
                 (LeagueTag.POR, SeasonTag.S09_10)]

VALID_SEASONS = [(LeagueTag.SCO, SeasonTag.S13_14),
                 (LeagueTag.SWI, SeasonTag.S10_11),
                 (LeagueTag.POR, SeasonTag.S11_12),
                 (LeagueTag.ITA, SeasonTag.S13_14),
                 (LeagueTag.SPA, SeasonTag.S12_13),
                 (LeagueTag.NET, SeasonTag.S12_13),
                 (LeagueTag.POR, SeasonTag.S14_15),
                 (LeagueTag.ENG, SeasonTag.S11_12),
                 (LeagueTag.FRA, SeasonTag.S11_12),
                 (LeagueTag.ENG, SeasonTag.S10_11)]

TEST_SEASONS = [(LeagueTag.FRA, SeasonTag.S09_10),
                (LeagueTag.GER, SeasonTag.S08_09),
                (LeagueTag.BEL, SeasonTag.S12_13),
                (LeagueTag.ENG, SeasonTag.S08_09),
                (LeagueTag.GER, SeasonTag.S11_12),
                (LeagueTag.ENG, SeasonTag.S12_13),
                (LeagueTag.SWI, SeasonTag.S14_15),
                (LeagueTag.FRA, SeasonTag.S08_09),
                (LeagueTag.GER, SeasonTag.S10_11),
                (LeagueTag.SCO, SeasonTag.S09_10),
                (LeagueTag.SPA, SeasonTag.S15_16),
                (LeagueTag.SWI, SeasonTag.S13_14),
                (LeagueTag.GER, SeasonTag.S15_16),
                (LeagueTag.FRA, SeasonTag.S10_11),
                (LeagueTag.POR, SeasonTag.S13_14)]

SMALL_TRAIN_SEASONS = [(LeagueTag.NET, SeasonTag.S14_15)]

SMALL_VALID_SEASONS = [(LeagueTag.SCO, SeasonTag.S13_14)]

SMALL_TEST_SEASONS = [(LeagueTag.FRA, SeasonTag.S09_10)]


def make_dataset(sql_path, season_list):
    datasets = list()
    for league_tag, season_tag in season_list:
        d = SingleSeasonSingleLeague(sql_path, league_tag, season_tag)
        datasets.append(d)
    return data.ConcatDataset(datasets)

def make_train_set(sql_path):
    return make_dataset(sql_path, TRAIN_SEASONS)

def make_valid_set(sql_path):
    return make_dataset(sql_path, VALID_SEASONS)

def make_test_set(sql_path):
    return make_dataset(sql_path, TEST_SEASONS)

def make_small_train_set(sql_path):
    return make_dataset(sql_path, SMALL_TRAIN_SEASONS)

def make_small_valid_set(sql_path):
    return make_dataset(sql_path, SMALL_VALID_SEASONS)

def make_small_test_set(sql_path):
    return make_dataset(sql_path, SMALL_TEST_SEASONS)







