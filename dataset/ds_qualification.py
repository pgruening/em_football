from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time
import torch

ONE_HOT_DIM = 55


def get_data_loader(batch_size, num_workers, *, is_train, validation_after_date, shuffle=None):
    data = pd.read_csv('qualifying.csv')
    to_one_hot = ToOneHot(data)

    data['time'] = [to_time(row) for _, row in data.iterrows()]
    thres = to_time(validation_after_date)

    if is_train:
        data = data[data['time'] < thres]
    else:
        data = data[data['time'] >= thres]

    if shuffle is not None:
        shuffle = is_train

    return DataLoader(
        Matches(data, to_one_hot, switch_teams=is_train),
        batch_size=batch_size, num_workers=num_workers
    )


def get_one_hot():
    data = pd.read_csv('qualifying.csv')
    to_one_hot = ToOneHot(data)
    return to_one_hot


class Matches(Dataset):
    def __init__(self, data, to_one_hot, switch_teams=False):
        self.data = data
        self.toh = to_one_hot
        self.st = switch_teams

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.st and np.random.rand() < .5:
            x1 = self.toh(row['team_1'])
            x2 = self.toh(row['team_2'])

            y1 = row['g1']
            y2 = row['g2']
        else:
            x1 = self.toh(row['team_2'])
            x2 = self.toh(row['team_1'])

            y1 = row['g2']
            y2 = row['g1']

        x = np.concatenate([x1, x2])
        return torch.Tensor(x).float(), torch.Tensor([y1, y2]).long()

    def __len__(self):
        return self.data.shape[0]


class ToOneHot():
    def __init__(self, all_data):
        teams = list(all_data['team_1']) + list(all_data['team_2'])
        self.teams_ = sorted(list(set(teams)))
        self.n = len(self.teams_)

    def __call__(self, team):
        out = np.zeros(self.n)
        out[self.teams_.index(team)] = 1.
        return out

    def get_team(self, idx):
        return self.teams_[idx]


def to_time(row):
    return time.mktime(
        time.strptime(f'{row["day"]}/{row["month"]}/{row["year"]}', "%d/%m/%Y")
    )
