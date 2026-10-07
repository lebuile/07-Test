import gymnasium as gym
from agents.DQN_agent import DQNAgent
import torch

env = gym.make("CartPole-v1", render_mode="human")

agent = DQNAgent(env=env)
num_episodes = 10000
model = torch.load(f"../../agents/cartpole_DQN_model_{num_episodes}.pth")
agent.q.load_state_dict(model)
agent.epsilon = 0.0 # No more random actions - use what we've learned

obs, info = env.reset()
done = False
while not done:
    action = agent.get_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    obs = next_obs

env.close()
