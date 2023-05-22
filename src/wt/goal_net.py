"""
Goal and reward waypoint network implementation.

By Anonymous Authors
"""

import gym
from gym import spaces
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F

from wt import dataset, layers, step, util

class KForwardGoalNetwork(pl.LightningModule):
    def __init__(self, obs_dim, goal_dim, hidden_dim, max_T, recurrent = False,
                learning_rate=1e-3, batch_size=1024, reward = False):
        super().__init__()
        self.recurrent = recurrent 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim
        self.max_T = max_T
        self.reward = reward

        if self.reward:
            assert goal_dim == 2
        self.net = nn.Sequential(nn.Linear(obs_dim + goal_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, goal_dim))

    def forward(self, obs_goal):
        if self.reward:
            return obs_goal[..., -self.goal_dim:] + self.net(obs_goal)

        return obs_goal[..., :self.goal_dim] + self.net(obs_goal)

    def training_step(
        self,
        batch,
        batch_idx: int,
        log_prefix: str = "train",
    ) -> torch.Tensor:
        """Computes loss for a training batch."""
        obs_goal, var, mask = batch
        if self.reward:
            # average reward case 
            loss = F.mse_loss(self(obs_goal), var)
            # cumulative reward case
        else:
            loss = 0
            for i in range(obs_goal.shape[1] - self.max_T):
                loss = loss + F.mse_loss(self(obs_goal[:, i]), obs_goal[:, i + self.max_T, :self.goal_dim])

            loss = loss / (obs_goal.shape[1] - self.max_T)

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Computes loss for a validation batch."""
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer used by PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class Manager:
    def __init__(self, global_goal, epsilon = 0.01, K = 20, goal_columns = (0, 1)):
        self.global_goal = global_goal
        self.goal_columns = goal_columns
        self.epsilon, self.K = epsilon, K
        self._observations = []
        self._actions = []

    def update_obs(self, obs):
        self._observations.append(obs)
        self._observations = self._observations[-self.K:]

    def update_act(self, act):
        self._actions.append(act)
        self._actions = self._actions[-self.K + 1:]

    @property
    def goal(self):
        if self._stuttering(self._observations):
            return self.global_goal #proposed_goal.detach().numpy()[0]
        return self.global_goal
        
    @property
    def actions(self):
        return None if not self._actions else self._actions + [np.zeros_like(self._actions[-1])]

    @property
    def observations(self):
        if self._stuttering(self._observations):
            return self._observations# [-1:]
        return self._observations

    def _stuttering(self, observations):
        if len(observations) < self.K:
            return False
        observations = np.concatenate([o[None, self.goal_columns] for o in observations], axis = 0)
        velocity = np.linalg.norm(observations[1:] - observations[:-1], axis = -1)
        return np.mean(velocity) <= self.epsilon

class Manager:
    def __init__(self, global_goal, epsilon = 0.01, K = 20, goal_columns = (0, 1), goal_append = False):
        self.global_goal = self.original_goal = global_goal
        self.goal_columns = goal_columns
        self.epsilon, self.K = epsilon, K
        self._observations = []
        self._actions = []
        self._goals = []
        self.goal_append = goal_append
        self.t = 0

    def update_obs(self, obs):
        self._observations.append(obs)
        self._observations = self._observations[-self.K:]
        self.t += 1

    def update_goal(self, goal):
        if self.goal_append:
            self._goals.append(goal)
            self._goals = self._goals[-self.K:]
        else:
            self.global_goal = goal

    def update_act(self, act):
        self._actions.append(act)
        self._actions = self._actions[-self.K + 1:]

    def step_goal(self, lambd, thres):
        assert not self.goal_append
        # self.global_goal = thres / (1 + np.exp(-lambd * (self.t - 60))) + self.original_goal
        self.global_goal = np.minimum(lambd * self.global_goal, thres)
        #self.global_goal = np.minimum(lambd + self.global_goal, thres)

    @property
    def goal(self):
        if self.goal_append:
            return self._goals
        else:
            return [self.global_goal] * len(self._observations)
        
    @property
    def actions(self):
        return None if not self._actions else self._actions + [np.zeros_like(self._actions[-1])]

    @property
    def observations(self):
        if self._stuttering(self._observations):
            return self._observations# [-1:]
        return self._observations

    def _stuttering(self, observations):
        if len(observations) < self.K:
            return False
        observations = np.concatenate([o[None, self.goal_columns] for o in observations], axis = 0)
        velocity = np.linalg.norm(observations[1:] - observations[:-1], axis = -1)
        return np.mean(velocity) <= self.epsilon

class ManualGoalNetwork(nn.Module):
    LARGE_GOALS = torch.tensor([[12, 0], [12, 7], [0, 7], [4, 15], [0, 22], [20, 7], [20, 15], [20, 22], [12, 22], [12, 15], [20, 0],
                                [28, 0], [28, 7], [36, 0], [36, 7], [36, 15], [28, 15], [28, 22], [36, 24]])

    # LARGE_GOALS = torch.tensor([[0, 8], [12, 7], [20, 7], [20, 15], [28, 15], [28, 24], [36, 24]])

    def __init__(self, obs_dim, goal_dim, level = 'large', **unused):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.dummy_param = nn.Parameter(torch.zeros(1))
        if level == 'large':
            self.goals = ManualGoalNetwork.LARGE_GOALS.to(self.dummy_param.device)

    def forward(self, obs_goal):
        self.goals = self.goals.to(obs_goal.device)
        loc, global_goals = obs_goal[..., :self.goal_dim], obs_goal[..., -self.goal_dim:]
        goal_dist = torch.linalg.norm(loc.unsqueeze(-2) - self.goals.unsqueeze(0), dim = -1)
        sorted_idx = goal_dist.argsort(dim = -1)
        global_prox_cond = torch.linalg.norm(self.goals[sorted_idx] - global_goals.unsqueeze(-2), dim = -1) < torch.linalg.norm(loc - global_goals, dim = -1).unsqueeze(-1)
        selected = global_prox_cond.int().argmax(dim = -1)
        return self.goals[selected]

