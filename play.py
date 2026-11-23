import sys

import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

import MLPlayable as M

env = DummyVecEnv([M.make_tetris_env()])
model = DQN.load(sys.argv[1] if len(sys.argv) > 1 else "tetris_dqn_model.zip", env=env)
obs = env.reset()
while pygame.get_init():  # closing the window quits pygame
    env.envs[0].env.render_enabled = True  # reset_game re-runs __init__, which turns it off again
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
