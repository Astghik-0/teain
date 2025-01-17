import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FONT_SIZE = 25
CAR_SPEED = 5
THING_SPEED = 5

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()

        # Define action space (e.g., 0: noop, 1: left, 2: right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (car position, obstacle positions, dodged count, etc.)
        self.observation_space = spaces.Box(low=0, high=100000, shape=(5, ), dtype=np.float32)

        self.reward = 0


        # spaces.Dict({
        #     "car_x": spaces.Box(0, SCREEN_WIDTH, shape=(1,), dtype=np.float32),
        #     "car_y": spaces.Box(0, SCREEN_HEIGHT, shape=(1,), dtype=np.float32),
        #     "obstacles": spaces.Box(0, max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(2,), dtype=np.float32),
        #     "dodged": spaces.Discrete(1000)
        # })

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Racing Gym Environment')
        self.clock = pygame.time.Clock()

        self.car_img = pygame.image.load('car_right.png')
        self.car_img = pygame.transform.scale(self.car_img, (50, 100))  # Resize to match original dimensions

        self.thing_img = pygame.image.load('obstacle.png')
        self.thing_img = pygame.transform.scale(self.thing_img, (50, 50))  # Resize to match original dimensions

        self.background_img = pygame.image.load('background_inv.png')
        self.background_img = pygame.transform.scale(self.background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.reward = 0

        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state variables
        self.car_x = SCREEN_WIDTH // 2
        self.car_y = SCREEN_HEIGHT - 120

        self.thing_x = random.randint(0, SCREEN_WIDTH - 50)
        self.thing_y = -50

        self.dodged = 0

        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if action == 1:  # Move left
            self.car_x -= CAR_SPEED
        elif action == 2:  # Move right
            self.car_x += CAR_SPEED

        # Ensure car stays within screen bounds
        self.car_x = np.clip(self.car_x, 0, SCREEN_WIDTH - 50)

        # Move the obstacle
        self.thing_y += THING_SPEED

        # Check collision
        if self.car_y < self.thing_y + 50 and self.car_x + 50 > self.thing_x > self.car_x - 50:
            self.done = True
            self.reward = 0

        # Reset obstacle if it moves off the screen
        if self.thing_y > SCREEN_HEIGHT:
            self.thing_y = -50
            self.thing_x = random.randint(0, SCREEN_WIDTH - 50)
            self.dodged += 1
            self.reward += 10

        return self._get_obs(), self.reward, self.done, False, {}

    def render(self, mode="human"):
        self.screen.fill((0, 0, 0))

        # Draw car
        self.screen.blit(self.car_img, (self.car_x, self.car_y))

        # Draw obstacle
        self.screen.blit(self.thing_img, (self.thing_x, self.thing_y))

        # Draw score
        score_text = self.font.render(f"Dodged: {self.dodged}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

        return self.screen

    def _get_obs(self):
        return np.array([self.car_x,
              self.car_y,
              self.thing_x,
              self.thing_y,
              self.dodged])


    def close(self):
        pygame.quit()
        
