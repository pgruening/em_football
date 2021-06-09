from bs4 import BeautifulSoup
import re
import pandas as pd
from DLBio.helpers import MyDataFrame

# rules:
# https://www.kicker.de/elfmeterschiessen_in_der_gruppenphase_der_modus_bei_der_em-763787/artikel


def get_date(x, y):
    rgx = r'(.*)(\d\d).(\d\d).(\d\d)'
    match = re.match(rgx, x)

    hour, minute = y.split(':')

    return {
        'day': match.group(2),
        'month': match.group(3),
        'year': match.group(4),
        'hour': hour,
        'minute': minute
    }


def create_em_match_table():
    path = 'EM Spielplan 2021 chronologisch - Datum + Uhrzeit _ EM 2020.html'
    tables_ = get_tables(path)

    df_group = read_group_table(tables_[0])
    df_ko = read_ko_table(tables_[1])

    df_group.to_csv('group_games.csv')
    df_ko.to_csv('ko_games.csv')


def read_group_table(table):
    df = MyDataFrame()
    for entry in table.findAll('tr'):
        x = entry.findAll('td')
        if not x:
            continue

        # fourth entry is always '-:-'
        if x[3].text != '-:-':
            continue

        teams = x[2].text.split(' – ')
        tmp = get_date(x[0].text, x[1].text)

        tmp.update({'team1': teams[0], 'team2': teams[1]})
        df.update(tmp)

    return df.get_df()


def read_ko_table(table):
    df = MyDataFrame()
    for entry in table.findAll('tr'):
        x = entry.findAll('td')
        if not x:
            continue

        # fourth entry is always '-:-'
        if x[4].text != '-:-':
            continue

        game_type = x[0].text
        # remove weird characters
        place = re.sub('[^A-Za-z]+', '', x[5].text)

        teams = x[3].text.split(' – ')
        tmp = {'game': game_type, 'place': place}
        tmp.update(get_date(x[1].text, x[2].text))
        tmp.update({'team1': teams[0], 'team2': teams[1]})

        df.update(tmp)

    return df.get_df()


def create_3rd_place_table():
    path = 'Direkter Vergleich, beste Gruppendritte und Elfmeterschießen - der Turniermodus der EURO 2020 - EURO 2020 - Fußball - sportschau.de.html'
    tables_ = get_tables(path)
    table = tables_[1]
    df = MyDataFrame()
    for entry in table.findAll('tr'):
        x = entry.findAll('td')
        if not x:
            continue

        df.update({
            'best_3rd_places': x[0].text,
            'against_winner_B': x[1].text,
            'against_winner_C': x[2].text,
            'against_winner_E': x[3].text,
            'against_winner_F': x[4].text,
        })

    df.get_df().to_csv('3rd_places.csv')


def read_qualification_results():

    def get_date(x):
        months_ = ['Januar', 'Februar', 'M�rz', 'April', 'Mai', 'Juni', 'Juli',
                   'August', 'September', 'Oktober', 'November', 'Dezember']
        rgx = r'([A-Z][a-z]) (\d+) (.*) (\d+):(\d+)'
        match = re.match(rgx, x)
        return {
            'day': int(match.group(2)),
            'month': months_.index(match.group(3)) + 1,
            'year': 2020,
            'hour': int(match.group(4)),
            'minute': int(match.group(5)),
        }

    path = 'EM 2021 Qualifikation Spielplan - Programm, Wettbewerbe und Ergebnisse.html'
    table = get_tables(path)[0]

    df = MyDataFrame()
    for entry in table.findAll('tr'):
        x = entry.findAll('td')
        if len(x) < 2:
            continue

        if 'Gruppe' in x[1].text:
            # read first line
            tmp = get_date(x[0].text)
        else:
            # read second line
            tmp['team_1'] = x[0].text
            tmp['team_2'] = x[4].text
            tmp['g1'] = int(x[2].text.split('-')[0])
            tmp['g2'] = int(x[2].text.split('-')[1])

            df.update(tmp)
            tmp = None

    df = df.get_df()
    df.to_csv('qualifying.csv')


def get_tables(path):
    with open(path, 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')
        tables_ = soup.findAll('table')
    return tables_


if __name__ == '__main__':
    create_em_match_table()
    create_3rd_place_table()
    read_qualification_results()
