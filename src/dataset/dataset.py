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
import math
import random

from dataset.db_query import query_all_players, query_matches, query_teams, get_player_ids_from_match
from dataset.util import MatchCaches

"""

Loads the data for a given season and a given league (using pytorch dataset mechanism).

Queues all needed data from the SQL database and handles wrong field values.
Also uses caching mechanisms and dataframe indices to provide a better performance.

"""
class SingleSeasonSingleLeague(data.Dataset):
    """
    Holds all matches of given season of a given league ordered by match dates.
    """

    # Determines whether we use a padding for players/teams that could not be found
    USE_PLAYER_PADDING = True
    USE_TEAM_PADDING = True

    # Caching directories, in our case this caching is much faster than access over dataframe index
    PLAYER_CACHE = dict()
    TEAM_CACHE = dict()

    """
    Initializes the dataset by queuing the SQL database given by sqpath.
    """
    def __init__(self, sqpath, league_tag, season_tag, undersample_probs=(1.0, 1.0, 1.0), load_odds=False):
        'Initialization'
        self.league = league_tag
        self.season = season_tag
        self.sqpath = sqpath
        self.undersample_probs = undersample_probs
        self.load_odds = load_odds

        sqlload_start = time.time()
        teams = None
        players = None
        matches = None
        with sqlite3.connect(self.sqpath) as conn:
            teams = query_teams(conn)
            players = query_all_players(conn)
            matches = query_matches(conn, self.league, self.season)
        sqlload_end = time.time()

        # Group players
        players.set_index("player_api_id", inplace=True)
        teams.set_index("team_api_id", inplace=True)

        logging.debug(
            "time for loading sql data {}".format(sqlload_end - sqlload_start))

        self.length = len(matches.index)

        process_start = time.time()

        samples = [
            self._generate_sample(idx, teams, players, matches)
            for idx in range(self.length)
        ]

        buf = []
        for X, y in samples:
            keep_sample = self._undersample(y)
            if keep_sample:
                buf.append((X, y))

        self.samples = buf

        process_end = time.time()
        logging.debug(
            "time for processing data {}".format(process_end - process_start))

    # Simply the number of the samples
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    # Gets a single item with a given index.
    def __getitem__(self, index):
        return self.samples[index]


    """
    Generates a single sample with a given index and needs the teams, players and matches dataframes.
    """
    def _generate_sample(self, index, teams, players, matches):
        # Select sample
        match = matches.loc[index]
        player_ids = get_player_ids_from_match(matches, index)

        # match time
        match_time = match["date"]

        # get teams
        team_home = self.select_team(match["home_team_api_id"], match_time, teams)
        team_away = self.select_team(match["away_team_api_id"], match_time, teams)

        encoded_team_away = self.encode_team(team_away[1])
        encoded_team_home = self.encode_team(team_home[1])

        # get players
        players_home = self.select_players(player_ids[1], match_time, players)
        players_away = self.select_players(player_ids[0], match_time, players)

        encoded_players_home = []
        encoded_players_away = []

        for idd, player in players_home.items():
            encoded_players_home.append(self.encode_player(player[1]))
        for idd, player in players_away.items():
            encoded_players_away.append(self.encode_player(player[1]))

        if (len(players_home) < 11):
            for _ in range(11 - len(players_home)):
                encoded_players_home.append(self.encode_player(None))
        if (len(players_away) < 11):
            for _ in range(11 - len(players_home)):
                encoded_players_away.append(self.encode_player(None))

        # Load data and get label
        X = {
            "home_team_name": match["home_team_long_name"],
            "away_team_name": match["away_team_long_name"],
            "team_home": encoded_team_home,
            "team_away": encoded_team_away,
            "players_home": encoded_players_home,
            "players_away": encoded_players_away,
            "home_team_goal": match["home_team_goal"],
            "away_team_goal": match["away_team_goal"]
        }

        if self.load_odds:
            odds_ids = [
                "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD",
                "IWA", "LBH", "LBD", "LBA", "PSH", "PSD", "PSA", "WHH", "WHD",
                "WHA", "SJH", "SJD", "SJA", "VCH", "VCD", "VCA", "GBH", "GBD",
                "GBA"
            ]
            
            for o in odds_ids:
                X[o] = match[o]


        y = torch.zeros(3)
        if match["result"] == "home":
            y = torch.as_tensor([1, 0, 0])
        elif match["result"] == "draw":
            y = torch.as_tensor([0, 1, 0])
        elif match["result"] == "away":
            y = torch.as_tensor([0, 0, 1])
        else:
            raise Exception("expected either 'home', 'draw' or 'away'")

        return X, y

    def _undersample(self, y):
        def biased_coin(p):
            " p is prob for True "
            if random.random() < p:
                return True
            else:
                return False

        hwp, dp, awp = self.undersample_probs
        if y[0] == 1:
            return biased_coin(hwp)
        if y[1] == 1:
            return biased_coin(dp)
        if y[2] == 1:
            return biased_coin(awp)


    """
        Selects a team with a given id (team_id) from a dataframe (teams).
    Uses internal caching to avoid a lot of searches against the dataframe.
    
    If there are multiple matches it selects the one which is the closest to the given match time.
    """
    def select_team(self, team_id, match_time, teams):
        if (math.isnan(team_id)):
            return [None, None]
        if (team_id in self.TEAM_CACHE):
            return self.TEAM_CACHE[team_id]
        if (team_id not in teams.index):
            return [None, None]

        team = teams.loc[team_id]
        sel_team = None

        if isinstance(team, pd.Series):
            self.TEAM_CACHE[team_id] = [None, team]
            return self.TEAM_CACHE[team_id]

        if (team.size > 0):
            sel_team = min(
                team.iterrows(),
                key=lambda t: self.time_diff(t[1]["date"], match_time))
            self.TEAM_CACHE[team_id] = sel_team
        elif self.USE_TEAM_PADDING:
            sel_team = [None, None]

        return sel_team

    """
	Selects a set of players with given ids (player_ids) from a dataframe (players).
	Also uses internal caching to avoid a lot of searches against the dataframe.
	
	If there are multiple matches it selects the one which is the closest to the given match time.
    """
    def select_players(self, player_ids, match_time, players):
        team_dict = {}
        for player_id in player_ids:
            p = self.select_player(player_id, match_time, players)
            if p is not None:
                team_dict[player_id] = p
            elif self.USE_PLAYER_PADDING:
                team_dict[player_id] = [None, None] # necessary because the player vars are arrays
        return team_dict

    """
	Selects a single player with a given id from a dataframe.
	Also uses internal caching to avoid a lot of searches against the dataframe.
    """
    def select_player(self, player_id, match_time, players):
        if (math.isnan(player_id)):
            return None
        if (player_id in self.PLAYER_CACHE):
            return self.PLAYER_CACHE[player_id]

        res = players.loc[player_id]

        if len(res) == 0:
            logging.debug("player with id {} could not be found in players table".format(player_id))
            return None

        sel_player = min(
            res.iterrows(),
            key=lambda t: self.time_diff(t[1]["date"], match_time))
        self.PLAYER_CACHE[player_id] = sel_player
        return sel_player

    """
	Encodes a single player to a pytorch tensor.
    """
    def encode_player(self, player):
        # zero_to_hundred_values = ["overall_rating", "potential"]
        cols = [
            "overall_rating", "potential", "crossing",
            "finishing", "heading_accuracy", "short_passing", "volleys",
            "dribbling", "curve", "free_kick_accuracy", "long_passing",
            "ball_control", "acceleration", "sprint_speed", "agility",
            "reactions", "balance", "shot_power", "jumping", "stamina",
            "strength", "long_shots", "aggression", "interceptions",
            "positioning", "vision", "penalties", "marking", "standing_tackle",
            "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking",
            "gk_positioning", "gk_reflexes"
        ]

        t = []
        if (player is None):
            t =  torch.zeros([35], dtype=torch.float32)
        else:
            t = torch.tensor(player[cols].astype("float32").values)
            t[torch.isnan(t)] = 0.0

        return t

    """
	Encodes a single team to a pytorch tensor.
    """
    def encode_team(self, team):
        cols = [
            "buildUpPlaySpeed",
            "buildUpPlayPassing",
            "chanceCreationPassing",
            "chanceCreationCrossing",
            "chanceCreationShooting",
            "defencePressure",
            "defenceAggression",
            "defenceTeamWidth"
        ]

        t = []
        if (team is not None):
            t = torch.tensor(team[cols].astype("float32").values)
            t[torch.isnan(t)] = 0.0
        else:
            t = torch.zeros([8], dtype=torch.float32)
        return t

    # help functions
    def time_diff(self, field, reference):
        return abs(pd.Timedelta(field - reference).total_seconds())

    def time_diff_nabs(self, field, reference):
        return pd.Timedelta(field - reference).total_seconds()


class SingleSeasonSingleLeagueTimeSlices(data.Dataset):
    """
    Like SingleSeasonSingleLeague holds all matches of a given season
    of a given league where __getitem__(i) returns the i-th match
    (sorted by date) but also the last @slice_size matches of both teams, i.e.
    a timeslices w.r.t. to the teams past performance.

    Note: If a team, at a timestep t, has played less than @slice_size matches 
    then we also return less than @slice_size matches.
    """

    def __init__(self, sqpath, league_tag, season_tag, slice_size, undersample_probs=(1.0, 1.0, 1.0), only_results=True):
        self.slice_size = slice_size
        self.samples = SingleSeasonSingleLeague(sqpath, league_tag, season_tag, undersample_probs=undersample_probs)
        self.only_results = only_results
        self.time_slices = list()
        self._create_slices()

    def _create_slices(self):
        match_caches = MatchCaches(self.slice_size)

        for i, (X, y) in enumerate(self.samples):
            away_team = X["away_team_name"]
            away_team_idxs = match_caches.get(away_team)
            match_caches.insert(away_team, i)

            home_team = X["home_team_name"]
            home_team_idxs = match_caches.get(home_team)
            match_caches.insert(home_team, i)

            self.time_slices.append((away_team_idxs, home_team_idxs))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def _collect(self, matches, teamname):
        results = list()
        for m, _ in matches:
            atn = m["away_team_name"]
            htn = m["home_team_name"]

            htg = float(m["home_team_goal"])
            atg = float(m["away_team_goal"])

            if teamname == atn:
                t = (atg, htg)
            elif teamname == htn:
                t = (htg, atg)
            else:
                raise Exception("teamname is not contained in match history ...")

            t = torch.tensor(t)
            results.append(t)
        return results

    def _match_and_hist(self, sample):
        match, result = sample["match"]
        ht_name = match["home_team_name"]
        at_name = match["away_team_name"]

        at_matches = sample["away_past_matches"]
        at_results = self._collect(at_matches, at_name)

        ht_matches = sample["home_past_matches"]
        ht_results = self._collect(ht_matches, ht_name)

        match["home_team_history"] = ht_results
        match["away_team_history"] = at_results
        return match, result



    def __getitem__(self, index):
        match = self.samples[index]
        away_ts, home_ts = self.time_slices[index]

        away_past_matches = list()
        for t in away_ts:
            pmatch = self.samples[t]
            away_past_matches.append(pmatch)

        home_past_matches = list()
        for t in home_ts:
            pmatch = self.samples[t]
            home_past_matches.append(pmatch)

        item = {
            "match": match,
            "away_past_matches": away_past_matches,
            "home_past_matches": home_past_matches
        }

        if self.only_results:
            m, r = self._match_and_hist(item)
            item = (m, r)

        return item
