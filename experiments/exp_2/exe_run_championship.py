from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset.ds_bet_and_win import get_one_hot
from DLBio.helpers import load_json
from DLBio.pytorch_helpers import get_device
from game_plan_api import get_winner, run
from helpers import load_model
from tqdm import tqdm
PATH = 'experiments/exp_2/exp_data/0009'


class MyModel():
    def __init__(self, max_lk):
        device = 'cpu'
        options = load_json(join(PATH, 'opt.json'))
        device = get_device()
        model = load_model(
            options, device, new_model_path=join(PATH, 'model.pt'),
            map_location=torch.device(device)
        )
        to_one_hot = get_one_hot()
        model.to_one_hot = to_one_hot
        model.d = device
        self.model = model
        self.max_lk = max_lk

    def __call__(self, team_1, team_2, no_draws=False):
        with torch.no_grad():
            out = self.model.predict(
                team_1, team_2, no_draws=no_draws, max_lk=self.max_lk)
        return out


def main():
    model = MyModel(True)
    run(model, 'em_2.md')


def winner_dist():
    model = MyModel(False)
    winners_ = []
    for _ in tqdm(range(100)):
        winner = get_winner(model)
        winners_.append(winner)

    plt.hist(np.array(winners_))
    plt.savefig('winner_dist_2.png')


if __name__ == '__main__':
    main()
    winner_dist()
