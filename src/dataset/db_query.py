import numpy as np
import pandas as pd
import sqlite3
import logging

from enum import Enum, auto

class SeasonTag(Enum):
    S08_09 = auto()
    S09_10 = auto()
    S10_11 = auto()
    S11_12 = auto()
    S12_13 = auto()
    S13_14 = auto()
    S14_15 = auto()
    S15_16 = auto()
    
    @staticmethod
    def as_year(val):
        if val == SeasonTag.S08_09:
            return "2008"
        elif val == SeasonTag.S09_10:
            return "2009"
        elif val == SeasonTag.S10_11:
            return "2010"
        elif val == SeasonTag.S11_12:
            return "2011"
        elif val == SeasonTag.S12_13:
            return "2012"
        elif val == SeasonTag.S13_14:
            return "2013"
        elif val == SeasonTag.S14_15:
            return "2014"
        elif val == SeasonTag.S15_16:
            return "2015"
        else: 
            raise ValueError("{} not a valid SeasonTag".format(val))

    @staticmethod
    def as_str(val):
        if val == SeasonTag.S08_09:
            return "2008/2009"
        elif val == SeasonTag.S09_10:
            return "2009/2010"
        elif val == SeasonTag.S10_11:
            return "2010/2011"
        elif val == SeasonTag.S11_12:
            return "2011/2012"
        elif val == SeasonTag.S12_13:
            return "2012/2013"
        elif val == SeasonTag.S13_14:
            return "2013/2014"
        elif val == SeasonTag.S14_15:
            return "2014/2015"
        elif val == SeasonTag.S15_16:
            return "2015/2016"
        else: 
            raise ValueError("{} not a valid SeasonTag".format(val))
        

class LeagueTag(Enum):
    """
    Enum value equals the primary key in the League table in the SQL database 
    for easier querying later ...
    """
    BEL = 1
    ENG = 1729
    FRA = 4769
    GER = 7809
    ITA = 10257
    NET = 13274
    POL = 15722
    POR = 17642
    SCO = 19694	
    SPA = 21518
    SWI = 24558


def query_teams(sql_conn):
    """
    We can't query teams based on league in this DB thus 
    we just load all teams (should not be to many anyway...)
    into a single DataFrame ...
    """

    qstr = """
        SELECT a.team_api_id, team_long_name, team_short_name, date, 
               buildUpPlaySpeed,
               buildUpPlaySpeedClass, 
               buildUpPlayDribbling,
               buildUpPlayDribblingClass, 
               buildUpPlayPassing,
               buildUpPlayPassingClass, 
               buildUpPlayPositioningClass,
               chanceCreationPassing, 
               chanceCreationPassingClass,
               chanceCreationCrossing, 
               chanceCreationCrossingClass,
               chanceCreationShooting, 
               chanceCreationShootingClass,
               chanceCreationPositioningClass, 
               defencePressure,
               defencePressureClass, 
               defenceAggression, 
               defenceAggressionClass,
               defenceTeamWidth, 
               defenceTeamWidthClass,
               defenceDefenderLineClass
        FROM Team_Attributes a
        LEFT JOIN Team t
        WHERE t.team_api_id = a.team_api_id
    """
    
    team_df = pd.read_sql_query(qstr, sql_conn)
    logging.info("got {} rows from db".format(team_df.shape[0])) 
    #logging.debug("no. of null values {} in teams table".format(teams_df.isnull().sum()))

    # convert the strings in column 'date' to pandas TimeStamp objects
    team_df["date"] = pd.to_datetime(team_df["date"], format='%Y-%m-%d %H:%M:%S.%f')
    
    return team_df


def query_matches(sql_conn, league_tag, season_tag):
    """
    Query all matches of a given league in a given season.
    """

    lid = league_tag.value
    sstr = SeasonTag.as_str(season_tag) 

    qstr = """
        SELECT date, country_id, league_id, season, match_api_id, 
               home_team_api_id, away_team_api_id, 
               Team1.team_long_name AS away_team_long_name, Team2.team_long_name AS home_team_long_name,
               Team1.team_short_name AS away_team_short_name, Team2.team_short_name AS home_team_short_name,
               home_team_goal, away_team_goal,
               home_player_1, home_player_2, 
               home_player_3, home_player_4, 
               home_player_5, home_player_6, 
               home_player_7, home_player_8, 
               home_player_9, home_player_10, 
               home_player_11, 
               away_player_1, away_player_2, 
               away_player_3, away_player_4, 
               away_player_5, away_player_6, 
               away_player_7, away_player_8, 
               away_player_9, away_player_10, 
               away_player_11,
               home_player_X1, home_player_X2,
               home_player_X3, home_player_X4,
               home_player_X5, home_player_X6,
               home_player_X7, home_player_X8,
               home_player_X9, home_player_X10,
               home_player_X11,
               home_player_Y1, home_player_Y2,
               home_player_Y3, home_player_Y4,
               home_player_Y5, home_player_Y6,
               home_player_Y7, home_player_Y8,
               home_player_Y9, home_player_Y10,
               home_player_Y11,
               away_player_X1, away_player_X2,
               away_player_X3, away_player_X4,
               away_player_X5, away_player_X6,
               away_player_X7, away_player_X8,
               away_player_X9, away_player_X10,
               away_player_X11,
               away_player_Y1, away_player_Y2,
               away_player_Y3, away_player_Y4,
               away_player_Y5, away_player_Y6,
               away_player_Y7, away_player_Y8,
               away_player_Y9, away_player_Y10,
               away_player_Y11
        FROM Match 
        INNER JOIN Team AS Team1 ON Team1.team_api_id = Match.away_team_api_id 
        INNER JOIN Team AS Team2 ON Team2.team_api_id = Match.home_team_api_id 
        WHERE league_id = {} AND season = '{}' 
    """.format(lid, sstr)
    
    matches_df = pd.read_sql_query(qstr, sql_conn)
    logging.info("got {} rows (league: {} (id {}), season: {}) from db".format(matches_df.shape[0], league_tag.name, lid, sstr))
    #logging.debug("no. of null values {} in matches table \n".format(matches_df.isnull().sum()))

    # convert the strings in column 'date' to pandas TimeStamp objects
    matches_df["date"] = pd.to_datetime(matches_df["date"], format='%Y-%m-%d %H:%M:%S.%f')

    def match_result(ht_goals, at_goals):
        if ht_goals > at_goals:
            return "home"
        elif ht_goals == at_goals:
            return "draw"
        else:
            return "away"
    
    matches_df = matches_df.assign(result=np.vectorize(match_result)(
        matches_df["home_team_goal"],
        matches_df["away_team_goal"]
    ))
    
    return matches_df

def query_all_players(sql_conn):
     qstr = """
        SELECT date, a.player_fifa_api_id, 
               a.player_api_id, 
               player_name, 
               overall_rating, 
               potential, 
               attacking_work_rate, 
               defensive_work_rate, 
               crossing, 
               finishing, 
               heading_accuracy, 
               short_passing, 
               volleys, 
               dribbling, 
               curve, 
               free_kick_accuracy, 
               long_passing, 
               ball_control, 
               acceleration, 
               sprint_speed, 
               agility, 
               reactions, 
               balance, 
               shot_power, 
               jumping, 
               stamina, 
               strength, 
               long_shots, 
               aggression, 
               interceptions, 
               positioning, 
               vision, 
               penalties, 
               marking, 
               standing_tackle, 
               sliding_tackle, 
               gk_diving, 
               gk_handling, 
               gk_kicking, 
               gk_positioning, 
               gk_reflexes, 
               birthday
        FROM Player_Attributes a
        LEFT JOIN Player t
        WHERE t.player_api_id = a.player_api_id
    """

     player_df = pd.read_sql_query(qstr, sql_conn)
     logging.info("got {} rows from db".format(player_df.shape[0])) 
     #logging.debug("no. of null values {} in player table".format(player_df.isnull().sum()))

     # convert the strings in column 'date' to pandas TimeStamp objects
     player_df["date"] = pd.to_datetime(player_df["date"], format='%Y-%m-%d %H:%M:%S.%f')

     return player_df

def query_player(sql_conn, player_id, match_time):
    qstr = """
        SELECT date, a.player_fifa_api_id, 
               a.player_api_id, 
               player_name, 
               overall_rating, 
               potential, 
               attacking_work_rate, 
               defensive_work_rate, 
               crossing, 
               finishing, 
               heading_accuracy, 
               short_passing, 
               volleys, 
               dribbling, 
               curve, 
               free_kick_accuracy, 
               long_passing, 
               ball_control, 
               acceleration, 
               sprint_speed, 
               agility, 
               reactions, 
               balance, 
               shot_power, 
               jumping, 
               stamina, 
               strength, 
               long_shots, 
               aggression, 
               interceptions, 
               positioning, 
               vision, 
               penalties, 
               marking, 
               standing_tackle, 
               sliding_tackle, 
               gk_diving, 
               gk_handling, 
               gk_kicking, 
               gk_positioning, 
               gk_reflexes, 
               birthday
        FROM Player_Attributes a
        LEFT JOIN Player t
        WHERE t.player_api_id = a.player_api_id AND t.player_api_id = {}
    """.format(player_id)

    player_df = pd.read_sql_query(qstr, sql_conn)
    logging.info("got {} rows from db".format(player_df.shape[0])) 
    #logging.debug("no. of null values {} in player table".format(player_df.isnull().sum()))

    # convert the strings in column 'date' to pandas TimeStamp objects
    player_df["date"] = pd.to_datetime(player_df["date"], format='%Y-%m-%d %H:%M:%S.%f')

    return min(player_df.iterrows(), key=lambda t : time_diff(t[1]["date"], match_time))

def time_diff(field, reference):
    return abs(pd.Timedelta(field - reference).total_seconds())

def query_multiple_players(sql_conn, player_ids, match_time):
    team_dict = {}
    for player_id in player_ids:
        player_df = query_player(sql_conn, player_id, match_time)
        team_dict[player_id] = player_df
    return team_dict


def get_player_names(team_dict):
    id_to_name = {} 
    for pid, df in team_dict.items():
        name = df.loc[0, 'player_name']
        id_to_name[pid] = name
    return id_to_name


def get_player_ids_from_match(matches_df, row):
    """ 
    Helper function that extracts all away and home team player ids 
    from the matches DataFrame (from query_matches) in a certain row.

    Args:
        matches_df (pd.DataFrame): matches table
        row (int): row in the matches table

    Returns:
        (pd.Series, pd.Series): Tuple with away_player_ids, home_player_ids
    """

    hp_columns = [
        'home_player_1', 'home_player_2', 
        'home_player_3', 'home_player_4', 
        'home_player_5', 'home_player_6',
        'home_player_7', 'home_player_8', 
        'home_player_9', 'home_player_10', 
        'home_player_11'
    ]
    
    ap_columns = [
        'away_player_1', 'away_player_2', 
        'away_player_3', 'away_player_4', 
        'away_player_5', 'away_player_6',
        'away_player_7', 'away_player_8', 
        'away_player_9', 'away_player_10', 
        'away_player_11'
    ]

    return matches_df.loc[row, ap_columns], matches_df.loc[row, hp_columns]     



def collect_team_names(matches_df, team_df):
    """
    Given some matches in a DataFrame it may be useful to extract all id, name pairs
    for all teams that occur in it.
    """
    team_names = {}

    for _, t in matches_df.loc[:, ["away_team_api_id", "home_team_api_id"]].iterrows():
        tmp = get_team_names_from_match(t.away_team_api_id, t.home_team_api_id, team_df)
    
        for tid, tname in tmp.items():
            if not tid in team_names: 
                team_names[tid] = tname

    return team_names







