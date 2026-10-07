import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.env = env
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.n_actions = env.action_space.n
        self.q = QNet(self.obs_dim, self.n_actions)
        self.opt = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, obs) -> int:  ## Epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_vals = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q_vals, dim=1).item())

    def update(self, obs, action: int, reward: float, done: bool, next_obs):

        # Compute current Q values for all actions
        q_pred_all = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))  ## Current Q(s, ·)

        # Compute estimates of Q values for next states and pick max for calculating estimated discounted return over episode.
        # Calculations of Q values for next states are not part of update (only based on current state), so we use torch.no_grad()
        with torch.no_grad():
            next_q_all = self.q(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0))
            max_next_q = torch.max(next_q_all, dim=1).values.item()
            target_value = reward + (0.0 if done else self.gamma * max_next_q)

            q_target_all = q_pred_all.detach().clone()
            q_target_all[0, action] = target_value # set only the taken action to target value, rest remain same as predicted

        loss = self.criterion(q_pred_all, q_target_all)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
