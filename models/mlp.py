import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, feat_dim):
        super(Model, self).__init__()
        D = 256
        self.one_hot_to_feature = nn.Sequential(
            nn.Linear(input_dim // 2, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, feat_dim),
        )

        self.feature_to_goals = nn.Sequential(
            nn.Linear(1 * feat_dim, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, output_dim)
        )

        self.to_one_hot = None
        self.d = None

    def forward(self, x):
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]

        x1 = self.one_hot_to_feature(x1)
        x2 = self.one_hot_to_feature(x2)
        rand_z = torch.rand(x1.shape).float().to(x1.device)

        x = x1 - x2 + rand_z
        out = self.feature_to_goals(x)
        return 8. * torch.sigmoid(out)

    def predict(self, team_1, team_2, no_draws=False):
        assert self.to_one_hot is not None
        assert self.d is not None

        x1 = self.to_one_hot(team_1)
        x2 = self.to_one_hot(team_2)
        x = np.concatenate([x1, x2])
        teams = torch.Tensor(x).float().to(self.d).unsqueeze(0)

        y1, y2 = self._get_pred(teams)
        ctr = 0
        while no_draws and y1 == y2:
            ctr += 1
            if max(y1, y2) > 6:
                y1 = 0
                y2 = 0

            a, b = self._get_pred(teams)
            y1 += a
            y2 += b
            if ctr > 10:
                if np.random.rand() < .5:
                    y1 += 1
                else:
                    y2 += 1

        return y1, y2

    def _get_pred(self, teams):
        pred = self(teams).cpu().numpy().round().astype('int')[0, :]
        return pred[0], pred[1]
