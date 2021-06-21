import pandas as pd
from DLBio.helpers import MyDataFrame
import time
import datetime

PATH = 'em_results.csv'


def run():
    players = {
        'Manja': evaluate('manja.md'),
        'Philipp': evaluate('philipp.md'),
        'Felix': evaluate('felix.md'),
        'Hans-Peter': evaluate('hans_peter.md'),
    }

    df = MyDataFrame()
    for name, val in players.items():
        df.update({
            'name': name,
            'accuracy': (1. - (val['points'] == 0).mean()).round(2),
            'points': val['points'].sum(),
        })

    df = df.get_df().sort_values('points', ascending=False)
    print(df)

    # print(players["Hans-Peter"])


def evaluate(path):
    gt = pd.read_csv(PATH)
    gt['ts'] = [to_time(row) for _, row in gt.iterrows()]
    gt = gt.sort_values('ts', ascending=True)
    df = read_markdown(path)

    output = MyDataFrame()

    for i in range(gt.shape[0]):
        y = gt.iloc[i]
        x = df[df['team1'] == y['team1']]
        x = x[x['team2'] == y['team2']]
        x = x.iloc[0, :]

        tmp = x['result'].split('-')
        g1 = int(tmp[0])
        g2 = int(tmp[1])

        perfect_match = g1 == y['g1'] and g2 == y['g2']
        if perfect_match:
            points = 3
        elif (g1 > g2) and (y['g1'] > y['g2']):
            # right class
            points = 1
        elif (g2 > g1) and (y['g2'] > y['g1']):
            # right class
            points = 1
        elif (g2 == g1) and (y['g2'] == y['g1']):
            points = 1
        else:
            points = 0

        output.update({
            'team1': y['team1'],
            'team2': y['team2'],
            'prediction': x['result'],
            'result': f'{y["g1"]}-{y["g2"]}',
            'points': points
        })

    output = output.get_df()
    return output


def read_markdown(path):
    do_save = False
    md_table = []

    with open(path, 'r') as file:
        for line in file.readlines():
            if line:
                line = line.replace('\n', '')

            if line == "## All Matches":
                do_save = True
                continue

            if do_save and line != '':
                md_table.append(line)

    df = MyDataFrame()
    header = md_table[0].split('|')[1:-1]
    header = [key.strip() for key in header]
    for line in md_table[2:]:
        line = line.split('|')[1:-1]
        tmp = dict()
        for idx, key in enumerate(header):
            tmp[key] = line[idx].strip()

        df.update(tmp)

    df = df.get_df().iloc[:, 1:]
    return df


def to_time(row):
    dtime = datetime.datetime(
        year=2000 + row['year'], month=row['month'],
        day=row['day'], hour=row['hour'], minute=row['minute']
    )
    timestamp = dtime.timestamp()
    return timestamp


if __name__ == '__main__':
    run()
