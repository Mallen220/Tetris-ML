import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# --- GLOBALS & SHAPES (Copied from HumanPlayable.py) ---
s_width = 800
s_height = 700
play_width = 300  # 300 // 10 = 30 width per block
play_height = 600  # 600 // 20 = 20 height per block
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(self.shape)]
        self.rotation = 0

# --- HELPER FUNCTIONS ---

def create_grid(locked_positions={}):
    grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid

def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)
    return positions

def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

def get_shape():
    return Piece(5, 0, random.choice(shapes))

# --- ENVIRONMENT CLASS ---

class TetrisEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(TetrisEnv, self).__init__()

        # Actions: 0: Left, 1: Right, 2: Rotate, 3: Soft Drop, 4: Hard Drop
        self.action_space = spaces.Discrete(5)

        # Observation: Flattened Grid (200) + Current Piece Info (7+3) + Next Piece Info (7)
        # 200 grid cells (0 or 1)
        # 7 one-hot for current shape
        # 3 for x, y, rotation (normalized)
        # 7 one-hot for next shape
        # Total: 217
        self.observation_space = spaces.Box(low=0, high=1, shape=(217,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.locked_positions = {}
        self.grid = []
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.max_steps = 2000 # Prevent infinite loops

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.score = 0
        self.steps = 0

        if self.render_mode == 'human':
            self._init_pygame()

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        # Apply action
        # 0: Left, 1: Right, 2: Rotate, 3: Soft Drop, 4: Hard Drop

        moved = False
        if action == 0: # Left
            self.current_piece.x -= 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x += 1
        elif action == 1: # Right
            self.current_piece.x += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x -= 1
        elif action == 2: # Rotate
            self.current_piece.rotation += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.rotation -= 1
        elif action == 3: # Soft Drop
            self.current_piece.y += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
            else:
                reward += 0.1 # Small reward for faster play
        elif action == 4: # Hard Drop
            while valid_space(self.current_piece, self.grid):
                self.current_piece.y += 1
                reward += 0.2 # Reward for hard drop depth
            self.current_piece.y -= 1

        # Gravity (Always move down 1 step per frame unless we just hard dropped)
        if action != 4:
            self.current_piece.y += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
                # Lock piece
                self._lock_piece()

                # Check for clears
                cleared = self._clear_rows()
                reward += self._calculate_reward(cleared)

                # Spawn new piece
                self.current_piece = self.next_piece
                self.next_piece = get_shape()

                # Check game over
                if check_lost(self.locked_positions):
                    terminated = True
                    reward -= 100 # Penalty for losing
            else:
                # Piece moved down successfully
                pass
        else:
            # Hard drop already locked? No, we just moved it to bottom.
            # We need to lock it now.
            self._lock_piece()
            cleared = self._clear_rows()
            reward += self._calculate_reward(cleared)
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            if check_lost(self.locked_positions):
                terminated = True
                reward -= 100

        self.grid = create_grid(self.locked_positions)

        # Time limit
        if self.steps >= self.max_steps:
            truncated = True

        # Render
        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, {}

    def _lock_piece(self):
        for pos in convert_shape_format(self.current_piece):
            p = (pos[0], pos[1])
            self.locked_positions[p] = self.current_piece.color

    def _clear_rows(self):
        # Correct implementation of clear_rows
        # Returns number of cleared rows

        rows_cleared = 0
        new_locked = {}

        shift = 0
        # Iterate from bottom (19) to top (0)
        for i in range(19, -1, -1):
            row = self.grid[i]
            if (0, 0, 0) not in row:
                rows_cleared += 1
                shift += 1
            else:
                # Keep this row, but shift it down
                for x in range(10):
                    if self.grid[i][x] != (0, 0, 0):
                        new_locked[(x, i + shift)] = self.grid[i][x]

        self.locked_positions = new_locked
        return rows_cleared

    def _calculate_reward(self, lines_cleared):
        # Heuristics
        grid = self.grid

        aggregate_height = 0
        bumpiness = 0
        holes = 0

        heights = [0] * 10
        for x in range(10):
            for y in range(20):
                if grid[y][x] != (0, 0, 0):
                    heights[x] = 20 - y
                    break

        aggregate_height = sum(heights)

        for i in range(9):
            bumpiness += abs(heights[i] - heights[i+1])

        for x in range(10):
            block_found = False
            for y in range(20):
                if grid[y][x] != (0, 0, 0):
                    block_found = True
                elif block_found and grid[y][x] == (0, 0, 0):
                    holes += 1

        # Reward function
        # Based on literature (e.g., Dellacherie)
        # Weights can be tuned

        reward = 0

        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 100

        reward -= 0.5 * aggregate_height
        reward -= 0.5 * holes
        reward -= 0.2 * bumpiness

        return reward

    def _get_observation(self):
        # 1. Grid (20x10) -> 200
        grid_flat = []
        for y in range(20):
            for x in range(10):
                if self.grid[y][x] == (0,0,0):
                    grid_flat.append(0)
                else:
                    grid_flat.append(1)

        # 2. Current Piece Info
        # One-hot shape
        cur_shape_oh = [0]*7
        cur_shape_oh[shapes.index(self.current_piece.shape)] = 1

        # Position & Rotation (Normalized)
        cur_info = [
            self.current_piece.x / 10.0,
            self.current_piece.y / 20.0,
            (self.current_piece.rotation % 4) / 4.0
        ]

        # 3. Next Piece Info
        next_shape_oh = [0]*7
        next_shape_oh[shapes.index(self.next_piece.shape)] = 1

        obs = np.array(grid_flat + cur_shape_oh + cur_info + next_shape_oh, dtype=np.float32)
        return obs

    def _init_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Tetris ML')
            self.window = pygame.display.set_mode((s_width, s_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode == "human":
            self._init_pygame()

            # Draw
            self.window.fill((0, 0, 0))

            # Title
            font = pygame.font.SysFont('Arial', 60)
            label = font.render('Tetris ML', 1, (255, 255, 255))
            self.window.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))

            # Grid
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    pygame.draw.rect(self.window, self.grid[i][j], (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

            # Grid Lines
            sx = top_left_x
            sy = top_left_y
            for i in range(len(self.grid)):
                pygame.draw.line(self.window, (128, 128, 128), (sx, sy + i*block_size), (sx + play_width, sy + i*block_size))
                for j in range(len(self.grid[i])):
                    pygame.draw.line(self.window, (128, 128, 128), (sx + j*block_size, sy), (sx + j*block_size, sy + play_height))

            # Border
            pygame.draw.rect(self.window, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 4)

            # Current Piece (Ghost/Active)
            # Already in grid? No, current piece is separate until locked.
            # We need to draw it.
            piece_pos = convert_shape_format(self.current_piece)
            for i in range(len(piece_pos)):
                x, y = piece_pos[i]
                if y > -1:
                    pygame.draw.rect(self.window, self.current_piece.color, (top_left_x + x * block_size, top_left_y + y * block_size, block_size, block_size), 0)

            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    # Training Loop

    # Create environment
    env = TetrisEnv(render_mode=None) # Set to 'human' to watch, but slower
    env = Monitor(env)

    # Check Env
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    print("Environment verified. Starting training...")

    # Model
    model = DQN("MlpPolicy", env, verbose=1,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                target_update_interval=500,
                exploration_fraction=0.1,
                exploration_final_eps=0.05
               )

    # Train
    try:
        model.learn(total_timesteps=100000, log_interval=10)
    except KeyboardInterrupt:
        print("Training interrupted.")

    model.save("tetris_dqn_v2")
    print("Model saved as tetris_dqn_v2.zip")

    # Evaluate/Watch
    print("Evaluating...")
    env = TetrisEnv(render_mode='human')
    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    env.close()
