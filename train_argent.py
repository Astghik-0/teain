from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from env import RacingEnv
import gymnasium as gym
import imageio
import pygame 
from IPython.display import Image

class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super(RewardLoggerCallback, self).__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Record the reward of the current step
        self.rewards.append(self.locals["rewards"])
        return True

    gym.envs.registration.register(
    id='RacingEnv-v0',
    entry_point=RacingEnv,
)
        # Create the environment
env = make_vec_env(lambda: RacingEnv(), n_envs=1)

# Set up the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
timesteps = 800000
reward_callback = RewardLoggerCallback()
model.learn(total_timesteps=timesteps, callback=reward_callback)

# Step 5: Save the model
model_path = "racing_dqn_model"
model.save(model_path)


# Visualize the trained model
env = gym.make("RacingEnv")
obs, _ = env.reset()

frames = []

for _ in range(80000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    # Render the current state to a Pygame surface and capture it
    screen = env.render()
    frame = pygame.surfarray.array3d(screen)
    frame = frame.swapaxes(0, 1)
    frames.append(frame)

    if done:
        break

# Save the frames as a GIF
gif_path = "game.gif"
imageio.mimsave(gif_path, frames, fps=10)

# Output the GIF in Colab

Image(filename=gif_path)