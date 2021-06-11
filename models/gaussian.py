import numpy as np
import torch
import torch.nn as nn

MAX_LK = True


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
            nn.Linear(D, D // 2),
        )

        self.mean1 = nn.Sequential(
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, feat_dim)
        )
        self.std1 = nn.Sequential(
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, feat_dim)
        )

        self.mean2 = nn.Sequential(
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, feat_dim)
        )
        self.std2 = nn.Sequential(
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Linear(D, feat_dim)
        )

        self.to_one_hot = None
        self.d = None

    def forward(self, x, max_lk=False):
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]

        x1 = self.one_hot_to_feature(x1)
        x2 = self.one_hot_to_feature(x2)

        out1 = self._get_output(x1, x2, max_lk)
        out2 = self._get_output(x2, x1, max_lk).flip(-1)
        # enforce position invariance
        return .5 * (out1 + out2)

    def _get_output(self, x1, x2, max_lk):
        x = torch.cat([x1, x2], -1)

        mu1 = self.mean1(x)
        mu2 = self.mean2(x)
        sig1 = self.std1(x)
        sig2 = self.std2(x)

        if max_lk:
            out1 = (mu1).sum(-1)
            out2 = (mu2).sum(-1)
        else:
            gauss1 = torch.normal(0, 1, size=mu1.shape).to(mu1.device)
            gauss2 = torch.normal(0, 1, size=mu2.shape).to(mu2.device)
            out1 = (gauss1 * sig1 + mu1).sum(-1)
            out2 = (gauss2 * sig2 + mu2).sum(-1)

        # return 8. * torch.sigmoid(torch.stack([out1, out2], 1))
        return torch.relu(torch.stack([out1, out2], 1))

    def predict(self, team_1, team_2, no_draws=False, max_lk=False):
        assert self.to_one_hot is not None
        assert self.d is not None

        y1, y2 = self._get_pred(team_1, team_2, max_lk)

        ctr = 0
        while no_draws and y1 == y2:
            ctr += 1
            if max(y1, y2) > 6:
                y1 = 0
                y2 = 0

            # turn maximum likelihood of to get any result that is no draw
            a, b = self._get_pred(team_1, team_2, False)
            y1 += a
            y2 += b
            if ctr > 100:
                if np.random.rand() < .5:
                    y1 += 1
                else:
                    y2 += 1

        return y1, y2

    def _get_pred(self, team_1, team_2, max_lk):
        return self._to_int(self._forward(team_1, team_2, max_lk))

    def _to_int(self, z):
        pred = z.cpu().numpy().round().astype('int')[0, :]
        return pred[0], pred[1]

    def _forward(self, team_1, team_2, max_lk):
        x1 = self.to_one_hot(team_1)
        x2 = self.to_one_hot(team_2)
        x = np.concatenate([x1, x2])
        teams = torch.Tensor(x).float().to(self.d).unsqueeze(0)
        return self(teams, max_lk)
