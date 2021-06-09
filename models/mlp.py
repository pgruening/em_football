import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, feat_dim):
        super(Model, self).__init__()

        self.one_hot_to_feature = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

        self.feature_to_goals = nn.Sequential(
            nn.Linear(3 * feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x1 = x[:, :x.shape[1] // 2]
        x2 = x[:, x.shape[1] // 2:]

        x1 = self.one_hot_to_feature(x1)
        x2 = self.one_hot_to_feature(x2)
        rand_z = torch.rand(x1.shape).float().to(x1.device)

        x = torch.cat([x1, x2, rand_z], -1)
        return self.feature_to_goals(x)
