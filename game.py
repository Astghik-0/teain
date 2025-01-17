from env import RacingEnv
import pygame 
if __name__ == "__main__":
    env = RacingEnv()
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)
        env.render()

    env.close()