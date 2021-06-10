from os.path import join
from os import chdir

# chdir('../..')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dataset.ds_qualification import get_one_hot, get_data_loader
from DLBio.helpers import MyDataFrame, load_json
from DLBio.pytorch_helpers import get_device
from helpers import load_model

BASE = 'experiments/exp_1'
#PATH = 'experiments/exp_1/exp_data/0002'
PATH = 'experiments/exp_1/exp_data/0009'


def test():
    model, to_one_hot, device = _setup()
    model.to_one_hot = to_one_hot
    model.d = device

    a, b = model.predict('Deutschland', 'Italien')


def plot_data():
    df = pd.read_csv(join(BASE, 'data.csv'))

    for _, row in df.iterrows():
        plt.scatter(row['x'], row['y'])
        plt.text(row['x'], row['y'], row['name'])

    plt.savefig(join(BASE, 'plot.png'))


def create_data():
    model, to_one_hot, device = _setup()

    df = MyDataFrame()
    out = []
    for i in range(to_one_hot.n):
        x = np.zeros(to_one_hot.n)
        x[i] = 1.
        x = torch.Tensor(x).float().unsqueeze(0).to(device)
        z = model.one_hot_to_feature(x).cpu().numpy()[0, :]
        df.update({
            'x': z[0], 'y': z[1], 'name': to_one_hot.get_team(i)
        })

    df = df.get_df().to_csv(join(BASE, 'data.csv'))


def create_table():
    model, to_one_hot, device = _setup()
    data_loader = get_data_loader(
        256, 0, is_train=False,
        validation_after_date={'year': 1900, 'month': 1, 'day': 1}
    )

    df = MyDataFrame()
    for sample in data_loader:
        teams, goals = sample[0].to(device), sample[1].to(device)
        pred = model(teams).cpu().numpy().round().astype('int')

        x1 = list(teams[:, :teams.shape[1] // 2].max(-1)[1].cpu())
        x1 = [to_one_hot.get_team(x) for x in x1]
        x2 = list(teams[:, teams.shape[1] // 2:].max(-1)[1].cpu())
        x2 = [to_one_hot.get_team(x) for x in x2]
        for i in range(len(x1)):
            df.update({
                'team_1': x1[i],
                'team_2': x2[i],
                'real_result': f'{goals[i,0].item()}-{goals[i,1].item()}',
                'pred_result': f'{pred[i,0].item()}-{pred[i,1].item()}',
            })

    df = df.get_df()
    df.to_csv(join(BASE, 'predicted_results.csv'))


def _setup():
    options = load_json(join(PATH, 'opt.json'))
    device = get_device()
    model = load_model(options, device, new_model_path=join(PATH, 'model.pt'))
    to_one_hot = get_one_hot()
    return model, to_one_hot, device


if __name__ == '__main__':
    with torch.no_grad():
        # test()
        # create_data()
        create_table()

    # plot_data()

# %%
