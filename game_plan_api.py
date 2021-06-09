import pandas as pd
import config
from DLBio.helpers import MyDataFrame
import re


class Team():
    def __init__(self, name):
        self.name = name

        group = [k for k, v in config.GROUPS.items() if name in v]
        assert group
        self.group = group[0]

        self.points = 0.
        self.goals_scored = 0.
        self.goals_conceded = 0.

    def update(self, scored, conceded):
        self.goals_scored += scored
        self.goals_conceded += conceded
        if scored > conceded:
            self.points += 3
        elif scored == conceded:
            self.points = 1
        else:
            self.points = 0


class Group():
    def __init__(self, name, teams):
        self.name = name
        self.teams = [x for x in teams.values() if x.group == name]
        assert self.teams
        assert len(self.teams) == 4

    def get_table(self):
        df = MyDataFrame()
        for x in self.teams:
            df.update({
                'name': x.name,
                'diff': x.goals_scored - x.goals_conceded,
                'points': x.points,
                'goals': x.goals_scored
            })

        return df.get_df().sort_values(
            ['points', 'diff', 'goals'],
            ascending=False, ignore_index=True
        )


def ask_group(func):
    group_games = pd.read_csv('group_games.csv')

    teams = {}
    for tmp in config.GROUPS.values():
        teams.update({x: Team(x) for x in tmp})

    for _, row in group_games.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        a, b = func(team1, team2)
        teams[team1].update(a, b)
        teams[team2].update(b, a)

    groups_ = {k: Group(k, teams) for k in config.GROUPS.keys()}

    return {name: group.get_table() for name, group in groups_.items()}


def ask_ko(func, group_tables):
    ko_games = pd.read_csv('ko_games.csv')
    places_3rd = pd.read_csv('3rd_places.csv').set_index('best_3rd_places')
    played_games = []

    for _, row in ko_games.iterrows():
        x1 = row['team1']
        x2 = row['team2']
        game_type = row['game']

        if game_type == 'AF':
            team1, team2 = resolve_af(
                x1, x2, group_tables, places_3rd
            )
        else:
            team1, team2 = resolve_other(x1, x2, played_games)

        a, b = func(team1, team2)

        played_games.append({
            'team1': team1, 'team2': team2,
            'type': row['game'], 'place': row['place'], 'result': [a, b]
        })
        print(x1, x2)
        print(played_games[-1])

    xxx = 0


def resolve_af(x1, x2, group_tables, places_3rd):
    def get_team(x, pos):
        rgx = r'(\d)([A-Z])'
        match = re.match(rgx, x)
        group = match.group(2).lower()
        assert pos == int(match.group(1))
        return group_tables[group].iloc[pos, :]['name']

    def resolve_3rd_place(x1, x2, pos_1, pos_2, group_tables, places_3rd):
        a = x1 if pos_1 != 3 else x2

        rgx_group = r'(\d)([A-Z])'
        other_group = re.match(rgx_group, a).group(2)
        key = f'against_winner_{other_group}'

        tab_3 = get_3rd_team_list(group_tables)
        tmp = places_3rd.loc[tab_3]
        group_3rd = tmp[key]

        if pos_1 == 3:
            return get_team(group_3rd, pos_1), get_team(x2, pos_2)
        else:
            return get_team(x1, pos_1), get_team(group_3rd, pos_2)

    rgx_pos = r'(\d)(.*)'
    pos_1 = int(re.match(rgx_pos, x1).group(1))
    pos_2 = int(re.match(rgx_pos, x2).group(1))

    if pos_1 == 3 or pos_2 == 3:
        return resolve_3rd_place(x1, x2, pos_1, pos_2, group_tables, places_3rd)
    else:
        return get_team(x1, pos_1), get_team(x2, pos_2)


def resolve_other(x1, x2, played_games):
    def get_winner(game):
        if game['result'][0] > game['result'][1]:
            return game['team1']
        else:
            return game['team2']

    def resolve(entry, played_games):
        def get_type(x):
            rgx = r'Sieger (AF|VF|HF)(.*)'
            return re.match(rgx, x).group(1)

        if get_type(entry) == 'AF':
            # London 1; London 2; Amsterdam; ...
            rgx = r'Sieger AF ([A-Z][a-z]+)( \d|)'
            match = re.match(rgx, entry)

            game = [x for x in played_games if x['type'] == 'AF']
            assert len(game) == 8
            if match.group(1) != '':
                game = [x for x in game if x['place'] == match.group(1)]

            if match.group(2) != '':
                # There are two matches in london
                assert len(game) > 1
                index = int(match.group(2))
                game = game[index - 1]

            else:
                assert len(game) == 1
                game = game[0]

            return get_winner(game)

        else:
            game_type = get_type(entry)
            game = [x for x in played_games if x['type'] == game_type]

            if game_type == 'VF':
                assert len(game) == 4
            if game_type == 'HF':
                assert len(game) == 2

            index = int(re.match(r'Sieger (VF|HF) (\d)', entry).group(2)) - 1
            game = game[index]
            return get_winner(game)

    return resolve(x1, played_games), resolve(x2, played_games)


def get_3rd_team_list(group_tables):
    df = MyDataFrame()
    for group, table in group_tables.items():
        tmp = (dict(table.iloc[2]))
        tmp['group'] = group
        df.update(tmp)

    df = df.get_df().sort_values(
        ['points', 'diff', 'goals'],
        ascending=False, ignore_index=True)

    out = sorted(list(df[:4].group))
    out = [x.upper() for x in out]
    return ", ".join(out)


def rand_func(team1, team2, no_draws=False):
    import random
    a = random.choice(list(range(10)))
    b = random.choice(list(range(10)))

    while a == b and no_draws:
        a = random.choice(list(range(10)))
        b = random.choice(list(range(10)))

    return a, b


def _test():
    group_tables = ask_group(rand_func)

    def func(t1, t2): return rand_func(t1, t2, no_draws=True)
    ask_ko(func, group_tables)


if __name__ == '__main__':
    _test()
