
# coding: utf-8

# In[210]:


import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[211]:


import torch
from torchvision import transforms
from torch.utils import data
from datetime import datetime
import itertools

from dataset.db_query import *


# In[212]:


class FootballDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, sqpath, league_tags, season_tags):
        'Initialization'
        self.leagues = league_tags
        self.seasons = season_tags
        self.sqpath = sqpath
        self.__load__()
        
  def __load__(self):
        with sqlite3.connect(self.sqpath) as conn:
            self.teams = query_teams(conn)
            self.players = query_all_players(conn)
            self.matches = None
            
            for league in self.leagues:
                for season in self.seasons:
                    m = query_matches(conn, league, season)
                    if (self.matches is None):
                        self.matches = m
                    else:
                        self.matches.append(m)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.matches.index)

  def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        match = self.matches.loc[index]
        player_ids = get_player_ids_from_match(self.matches, index)
        
        # match time
        match_time = match["date"]
        
        # get teams
        team_home = self.teams.loc[self.teams['team_api_id'] == match["home_team_api_id"]]
        team_away = self.teams.loc[self.teams['team_api_id'] == match["away_team_api_id"]]
            
        team_home = min(team_home.iterrows(), key=lambda t : self.time_diff(t[1]["date"], match_time))
        team_away = min(team_away.iterrows(), key=lambda t : self.time_diff(t[1]["date"], match_time))
            
        # get players
        players_home = self.select_players(player_ids[1], match_time)
        players_away = self.select_players(player_ids[0], match_time)
            
        # get current trend
        trend_home = self.current_form(match["home_team_api_id"], match)
        trend_away = self.current_form(match["away_team_api_id"], match)
        
        #encode
        encoded_team_away = self.encode_team(team_away[1])
        encoded_team_home = self.encode_team(team_home[1])
            
        encoded_players_home = []
        encoded_players_away = []
            
        for idd, player in players_home.items():
            encoded_players_home.append(self.encode_player(player[1]))
        for idd, player in players_away.items():
            encoded_players_away.append(self.encode_player(player[1]))
        
        encoded_players_home = np.stack(encoded_players_home, axis = 0)
        encoded_players_away = np.stack(encoded_players_away, axis = 0)
        
        metadata = {
            'away_team' : match["away_team_long_name"],
            'home_team' : match["home_team_long_name"],
            'result' : match["result"],
            'home_team_goals' : match["home_team_goal"],
            'away_team_goals' : match["away_team_goal"],
            'date' : match["date"],
            'match_id' : match["match_api_id"]
        }

        # Load data and get label
        X = [encoded_team_home, encoded_team_away, encoded_players_home, encoded_players_away, trend_home, trend_away]
        
        # X is in the following form:
        # [TEAM_HOME, TEAM_AWAY, STACK[[PLAYER_1], [PLAYER_2] ... [PLAYER_11]], STACK[[PLAYER_1], [PLAYER_2] ... [PLAYER_11]]

        return X, match["result"], metadata
  def current_form(self, team_id, match, count = 5):
    matches_all = self.matches.loc[(self.matches['home_team_api_id'] == team_id) | (self.matches['away_team_api_id'] == team_id)]
    m_sorted = sorted(matches_all.iterrows(), key=lambda t : self.time_diff_nabs(t[1]["date"], match["date"]))
    m_filtered = filter(lambda x : self.time_diff_nabs(x[1]["date"], match["date"]) > 0, m_sorted)
    wins = 0
    for m in itertools.islice(m_filtered, 5):
        if (m[1]["result"] == "home" and m[1]["home_team_api_id"] == team_id):
            wins += 2
        elif (m[1]["result"] == "away" and m[1]["away_team_api_id"] == team_id):
            wins += 2
        elif (m[1]["result"] == "draw"):
            wins += 1
    
    return self.one_hot_value_int(wins, 0, count * 2)
  def select_players(self, player_ids, match_time):
    team_dict = {}
    for player_id in player_ids:
        team_dict[player_id] = self.select_player(player_id, match_time)
    return team_dict
    
  def select_player(self, player_id, match_time):
    res = self.players.loc[self.players['player_api_id'] == player_id]
    return min(res.iterrows(), key=lambda t : self.time_diff(t[1]["date"], match_time))
    
  def encode_player(self, player):
    zero_to_hundred_values = [
        "overall_rating",
        "potential"
    ]
    category_values = [
        "attacking_work_rate", 
        "defensive_work_rate",
        "crossing", 
        "finishing", 
        "heading_accuracy", 
        "short_passing", 
        "volleys", 
        "dribbling", 
        "curve", 
        "free_kick_accuracy", 
        "long_passing", 
        "ball_control", 
        "acceleration", 
        "sprint_speed", 
        "agility", 
        "reactions", 
        "balance", 
        "shot_power", 
        "jumping", 
        "stamina", 
        "strength", 
        "long_shots", 
        "aggression", 
        "interceptions", 
        "positioning", 
        "vision", 
        "penalties", 
        "marking", 
        "standing_tackle", 
        "sliding_tackle", 
        "gk_diving", 
        "gk_handling", 
        "gk_kicking", 
        "gk_positioning", 
        "gk_reflexes"
    ]
    
    vals = []
    for z_to_h in zero_to_hundred_values:
        vals.append(self.one_hot_value_int_digits(player[z_to_h], 3))
    for category in category_values:
        vals.append(self.one_hot_category(player[category], self.players[category].unique()))
    
    return vals
    
  def encode_team(self, team):
    zero_to_hundred_values = [
        "buildUpPlaySpeed",
        "buildUpPlayDribbling",
        "buildUpPlayPassing",
        "chanceCreationPassing",
        "chanceCreationCrossing",
        "chanceCreationShooting",
        "defencePressure",
        "defenceAggression",
        "defenceTeamWidth"
    ]
    category_values = [
        "buildUpPlaySpeedClass",
        "buildUpPlayDribblingClass",
        "buildUpPlayPassingClass",
        "buildUpPlayPositioningClass",
        "chanceCreationPassingClass",
        "chanceCreationCrossingClass",
        "chanceCreationShootingClass",
        "chanceCreationPositioningClass",
        "defencePressureClass",
        "defenceAggressionClass",
        "defenceTeamWidthClass",
        "defenceDefenderLineClass"
    ]
    
    vals = []
    for z_to_h in zero_to_hundred_values:
        vals.append(self.one_hot_value_int_digits(team[z_to_h], 3))
    for category in category_values:
        vals.append(self.one_hot_category(team[category], self.teams[category].unique()))
    
    return vals
    
  # help functions
  def time_diff(self, field, reference):
    return abs(pd.Timedelta(field - reference).total_seconds())
  def time_diff_nabs(self, field, reference):
    return pd.Timedelta(field - reference).total_seconds()

  # one hot functions
  def one_hot_category(self, value, categories):
    return self.one_hot_value_int(np.where(categories==value)[0], 0, len(categories) - 1)
  def one_hot_value_int(self, value, minv, maxv):
    return np.eye(maxv - minv + 1)[int(value - minv)]
  def one_hot_value_int_digits(self, value, digits):
    res = []
    digstr = str(int(value))
    for k in range(0, digits):
        digit = 0
        if (k < len(digstr)):
            digit = int(digstr[-(k + 1)])
        res.append(self.one_hot_value_int(digit, 0, 9))
    return np.stack(res, axis=0 )