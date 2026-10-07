import gymnasium as gym
from agents.DQN_agent import DQNAgent
import torch

env = gym.make("CartPole-v1", render_mode="human") ## If training over many episodes set render_mode to None

agent = DQNAgent(env=env)

num_episodes = 10

for i in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs

env.close()
torch.save(agent.q.state_dict(), f"../../agents/cartpole_DQN_model_{num_episodes}.pth")
