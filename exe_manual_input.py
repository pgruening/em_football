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


def main():
    run(manual_input, 'manja.md')


def manual_input(team1, team2, no_draws=False):
    while True:
        print(f'{team1}–{team2}?')
        x = input()
        try:
            a, b = x.split(' ')
        except:
            continue

        print(f'{a}–{b}? y/N')
        c = input()
        if c == 'y':
            return int(a), int(b)


if __name__ == '__main__':
    main()
