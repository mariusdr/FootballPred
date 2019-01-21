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

from dataset.db_query import query_all_players, query_matches, query_teams, get_player_ids_from_match
from dataset.util import MatchCaches


class SingleSeasonSingleLeague(data.Dataset):
    """
    Holds all matches of given season of a given league ordered by match dates.
    """
    USE_PLAYER_PADDING = False

    def __init__(self, sqpath, league_tag, season_tag):
        'Initialization'
        self.league = league_tag
        self.season = season_tag
        self.sqpath = sqpath

        sqlload_start = time.time()
        teams = None
        players = None
        matches = None
        with sqlite3.connect(self.sqpath) as conn:
            teams = query_teams(conn)
            players = query_all_players(conn)
            matches = query_matches(conn, self.league, self.season)
        sqlload_end = time.time()

        logging.debug(
            "time for loading sql data {}".format(sqlload_end - sqlload_start))

        self.length = len(matches.index)

        process_start = time.time()

        self.samples = [
            self._generate_sample(idx, teams, players, matches)
            for idx in range(self.length)
        ]

        process_end = time.time()
        logging.debug(
            "time for processing data {}".format(process_end - process_start))

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        return self.samples[index]

    def _generate_sample(self, index, teams, players, matches):
        # Select sample
        match = matches.loc[index]
        player_ids = get_player_ids_from_match(matches, index)

        # match time
        match_time = match["date"]

        # get teams
        team_home = teams.loc[teams['team_api_id'] ==
                              match["home_team_api_id"]]
        team_away = teams.loc[teams['team_api_id'] ==
                              match["away_team_api_id"]]

        team_home = min(
            team_home.iterrows(),
            key=lambda t: self.time_diff(t[1]["date"], match_time))
        team_away = min(
            team_away.iterrows(),
            key=lambda t: self.time_diff(t[1]["date"], match_time))

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

        # Load data and get label
        X = {
            "home_team_name": match["home_team_long_name"],
            "away_team_name": match["away_team_long_name"],
            "team_home": encoded_team_home,
            "team_away": encoded_team_away,
            "players_home": encoded_players_home,
            "players_away": encoded_players_away,
        }

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

    # def current_form(self, team_id, match, count=5):
    # matches_all = self.matches.loc[
    # (self.matches['home_team_api_id'] == team_id) |
    # (self.matches['away_team_api_id'] == team_id)]
    # m_sorted = sorted(
    # matches_all.iterrows(),
    # key=lambda t: self.time_diff_nabs(t[1]["date"], match["date"]))
    # m_filtered = filter(
    # lambda x: self.time_diff_nabs(x[1]["date"], match["date"]) > 0,
    # m_sorted)
    # wins = 0
    # for m in itertools.islice(m_filtered, 5):
    # if (m[1]["result"] == "home"
    # and m[1]["home_team_api_id"] == team_id):
    # wins += 2
    # elif (m[1]["result"] == "away"
    # and m[1]["away_team_api_id"] == team_id):
    # wins += 2
    # elif (m[1]["result"] == "draw"):
    # wins += 1

    # return self.one_hot_value_int(wins, 0, count * 2)

    def select_players(self, player_ids, match_time, players):
        team_dict = {}
        for player_id in player_ids:
            p = self.select_player(player_id, match_time, players)
            if p is not None:
                team_dict[player_id] = p
            elif self.USE_PLAYER_PADDING:
                team_dict[player_id] = [None, None] # necessary because the player vars arr arrays
        return team_dict

    def select_player(self, player_id, match_time, players):
        res = players.loc[players['player_api_id'] == player_id]

        if len(res) == 0:
            logging.debug("player with id {} could not be found in players table".format(player_id))
            return None

        return min(
            res.iterrows(),
            key=lambda t: self.time_diff(t[1]["date"], match_time))

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
            t =  torch.zeros([35], dtype=torch.int32)
        else:
            t = torch.tensor(player[cols].astype("float32").values)
            t[torch.isnan(t)] = 0.0
        
        return t

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
        t = torch.tensor(team[cols].astype("float32").values)
        t[torch.isnan(t)] = 0.0
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

    def __init__(self, sqpath, league_tag, season_tag, slice_size):
        self.slice_size = slice_size
        self.samples = SingleSeasonSingleLeague(sqpath, league_tag, season_tag)

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
        return item
