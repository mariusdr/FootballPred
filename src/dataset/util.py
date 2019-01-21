from collections import deque

class MatchCaches(object):
    """
    We use this cache as a kind of 'sliding window' over the 
    past couple matches each of team to track their recent performance.
    
    For each observed team we save @num_of_cached_matches matches in a 
    FIFO queue.
    """
    def __init__(self, num_matches_saved):
        self.num_matches_saved = num_matches_saved
        self.queues = {}

    def insert(self, team_name, match):
        if team_name not in self.queues:
            self.queues[team_name] = deque([match])
        else:
            # if the deque is full we need to evict one item on the left
            if len(self.queues[team_name]) == self.num_matches_saved:
                self.queues[team_name].popleft()

            self.queues[team_name].append(match)

    def contains(self, team_name):
        if team_name not in self.queues:
            return False
        if len(self.queues[team_name]) > 0:
            return True
        return False

    def get(self, team_name):
        if not self.contains(team_name):
            return []
        return list(self.queues[team_name])


def pretty_print_result(y):
    if y[0] == 1:
        return "home win"
    if y[1] == 1:
        return "draw"
    if y[2] == 1:
        return "away win"

def pretty_print_match(match):
    X, y = match
    hn = X["home_team_name"]
    an = X["away_team_name"]

    hps = X["players_home"]
    aps = X["players_away"]

    r = pretty_print_result(y)
    print(
        "{} (home) vs. {} (away) | Result: {} | num. of home player entries: {} | num. of away player entries: {}"
        .format(hn, an, r, len(hps), len(aps)))


def pretty_print_match_ts(sample):
    match = sample["match"]
    pretty_print_match(match)

    print("home team past matches:")
    h_matches = sample["home_past_matches"]
    for m in h_matches:
        pretty_print_match(m)

    print("away team past matches:")
    a_matches = sample["away_past_matches"]
    for m in a_matches:
        pretty_print_match(m)
