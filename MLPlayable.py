import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

from pygame import surface
from pygame.examples import grid
from stable_baselines3.common.monitor import Monitor

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

def load_high_score():
    if os.path.exists('scores.txt'):
        with open('scores.txt', 'r') as f:
            return int(f.read().strip())
    return 0

HIGH_SCORE = load_high_score()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

# SHAPE FORMATS

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


# index 0 - 6 represent shape


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(self.shape)]
        self.rotation = 0


def get_shape():  # TODO make this go through all shapes instead of true random
    return Piece(5, 0, random.choice(shapes))

def create_grid(locked_positions={}):
    grid = [[(0, 0,0 ) for x in range(10)] for y in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                color = locked_positions[(j, i)]
                grid[i][j] = color

    return grid

class TetrisEnv:
    def __init__(self, window_id=None):
        pygame.init()

        self.window_id = window_id

        self.win = pygame.display.set_mode((s_width, s_height))
        self.run = True
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.27  # controls how fast the pieces fall
        self.level_time = 0
        self.score = 0
        self.change_piece = False

        self.window = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption('Tetris AI')
        self.clock = pygame.time.Clock()
        self.render_enabled = True  # toggle this to False if running headless
        self.fast_mode = True
        self.done = check_lost(self.locked_positions)

        if self.render_enabled:
            pygame.init()
            # Create unique window position based on window_id
            if window_id is not None:
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{100 + (window_id % 2) * 400},{100 + (window_id // 2) * 300}"
            self.window = pygame.display.set_mode((s_width, s_height))
            pygame.display.set_caption(f'Tetris AI - Window {window_id}' if window_id else 'Tetris AI')

        # self.step(self, )

    def get_grid(self):
        return self.grid

    def get_state(self):
        flat_grid = [1 if cell != (0, 0, 0) else 0 for row in self.grid for cell in row]
        piece_info = self.get_piece_info()
        state_vector = np.array(flat_grid + piece_info, dtype=np.float32)
        return state_vector

    def lock_piece(self):
        for pos in convert_shape_format(self.current_piece):
            if not (0 <= pos[0] < 10 and 0 <= pos[1] < 20):
                continue
            if pos[1] > -1:
                self.locked_positions[(pos[0], pos[1])] = self.current_piece.color

        self.current_piece = self.next_piece
        self.next_piece = get_shape()
        self.grid = create_grid(self.locked_positions)
        cleared = clear_rows(self.grid, self.locked_positions)
        self.score += cleared * 10
        self.done = check_lost(self.locked_positions)
        reward = self.evaluate_board(self.grid, cleared)

        return reward

    def step(self, action):

        self.fall_time += pygame.time.get_ticks()
        self.level_time += pygame.time.get_ticks()
        self.render()

        if self.level_time / 1000 >= 5:  # every 5 seconds increase the fall speed
            # self.level_time = 0
            if self.fall_speed > 0.12:  # minimum fall speed
                self.fall_speed -= 0.005  # decrease the fall speed

        if self.fall_time / 1000 >= self.fall_speed:
            self.fall_time = 0
            self.current_piece.y += 1
            if not(valid_space(self.current_piece, self.grid)):
                self.current_piece.y -= 1
                self.change_piece = True

        if action == 4:  # "Best move"
            target_rotation, target_x = self.get_best_action()

            # First priority: rotation
            if self.current_piece.rotation % len(self.current_piece.shape) != target_rotation:
                self.do_action(3)  # rotate
            # Then adjust horizontal position
            elif self.current_piece.x < target_x:
                self.do_action(1)  # move right
            elif self.current_piece.x > target_x:
                self.do_action(0)  # move left
            # Else: do nothing — wait to fall naturally or with soft drop
        else:
            self.do_action(action)

        # Drop piece down one row
        self.current_piece.y += 1
        if not valid_space(self.current_piece, self.grid):
            self.current_piece.y -= 1
            for pos in convert_shape_format(self.current_piece): #Lockes the piece in place
                if pos[1] > -1:
                    self.locked_positions[(pos[0], pos[1])] = self.current_piece.color

            self.current_piece = self.next_piece
            self.next_piece = get_shape()

        # Update grid and score
        self.grid = create_grid(self.locked_positions)

        ### REWARD MATH
        self.done = check_lost(self.locked_positions)

        cleared = clear_rows(self.grid, self.locked_positions)
        self.score += cleared * 10

        reward = self.evaluate_board(self.grid, cleared)

        # print(reward, self.score, cleared, count_holes(self.grid), get_piece_height(self.current_piece))
        return self.get_grid(), reward, self.done

    def reset_game(self):
        self.__init__()
        return self.get_state()

    def do_action(self, action):
        if action == 0:
            self.current_piece.x -= 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x += 1
        elif action == 1:
            self.current_piece.x += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x -= 1
        elif action == 2:
            self.current_piece.y += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
        elif action == 3:
            self.current_piece.rotation += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.rotation -= 1

    def get_piece_info(self):
        current_one_hot = [0] * len(shapes)
        current_index = shapes.index(self.current_piece.shape)
        current_one_hot[current_index] = 1

        next_one_hot = [0] * len(shapes)
        next_index = shapes.index(self.next_piece.shape)
        next_one_hot[next_index] = 1

        x = self.current_piece.x / 10
        y = self.current_piece.y / 20
        rotation = self.current_piece.rotation / 4

        return current_one_hot + [x, y, rotation] + next_one_hot

    def close(self):
        if self.render_enabled:
            pygame.quit()

    def render(self):
        if not self.render_enabled:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self.render_enabled = False
                return

        global HIGH_SCORE
        if self.score > HIGH_SCORE:
            HIGH_SCORE = self.score
            with open('scores.txt', 'w') as f:
                f.write(str(HIGH_SCORE))

        draw_window(self.window, self.grid, score=self.score, high_score=HIGH_SCORE)
        draw_next_shape(self.next_piece, self.window)
        pygame.display.update()
        if self.fast_mode:
            self.clock.tick(1000)
        else:
            self.clock.tick(20)

    def get_best_action(self):
        best_value = float('-inf')
        best_action = None

        for rotation in range(4):
            for x in range(-2, 12):
                test_piece = Piece(self.current_piece.x, self.current_piece.y, self.current_piece.shape)
                test_piece.rotation = rotation
                test_piece.x = x

                while valid_space(test_piece, self.grid):
                    test_piece.y += 1
                test_piece.y -= 1

                locked_copy = self.locked_positions.copy()
                for pos in convert_shape_format(test_piece):
                    if pos[1] > -1:
                        locked_copy[(pos[0], pos[1])] = test_piece.color

                grid_copy = create_grid(locked_copy)
                cleared = clear_rows(grid_copy, locked_copy)

                value = self.evaluate_board(grid_copy, cleared)

                if value > best_value:
                    best_value = value
                    best_action = (rotation, x)

        return best_action

    def evaluate_board(self, grid, cleared=0):
        """Evaluate the quality of a board position""" #TODO modify this function to include more heuristics
        """Returns Reward Value"""
        # Calculate height metrics
        heights = []
        aggregate_height = 0
        for col in range(10):
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    height = 20 - row
                    heights.append(height)
                    aggregate_height += height
                    break
            else:
                heights.append(0)

        max_height = max(heights) if heights else 0

        # Calculate bumpiness
        bumpiness = 0
        for i in range(9):
            bumpiness += abs(heights[i] - heights[i + 1])

        # Count holes
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block:
                    holes += 1

        # Reward clearing - higher reward for lower clears
        clear_reward = 0
        if cleared > 0:
            # Base reward + bonus for multiple lines
            clear_reward = 100 * cleared + (100 if cleared >= 4 else 0)
            # Height bonus: more reward for lower clears
            height_bonus = max(0, (15 - max_height) * 1)
            clear_reward += height_bonus

        # Row fill reward: reward rows that are nearly full
        row_fill_reward = 0
        for row in grid:
            filled = sum(1 for cell in row if cell != (0, 0, 0))
            if 0 < filled < 10:  # ignore empty and cleared rows
                # Scale the reward quadratically to favor fuller rows
                row_fill_reward += (filled / 10) ** 2 * 6  # You can tune the weight (5)

        time_running = self.clock.get_time()
        time_reward = time_running * 10

        # Penalties (make these dominant factors)
        height_penalty = max_height * -10  # Strong penalty for max height
        aggregate_height_penalty = aggregate_height * -0.2  # Penalty for total height
        hole_penalty = holes * -1  # Significant hole penalty
        bumpiness_penalty = bumpiness * -5  # Penalize uneven surfaces


        # Lost game penalty
        lost_penalty = -1000 if self.done else 0

        # Survival bonus (small to encourage longevity without promoting height)
        survival_bonus = 0.1 if not self.done else 0

        # Calculate total value
        value = (
                clear_reward +
                # height_penalty +
                row_fill_reward +
                time_reward +
                aggregate_height_penalty +
                # hole_penalty +
                bumpiness_penalty +
                survival_bonus +
                lost_penalty
        )

        # print("Value: ", value,
        #       "cleared_reward: ", clear_reward,
        #         "height_penalty: ", height_penalty,
        #         "row_fill_reward: ", row_fill_reward,
        #         "time_reward:", time_reward,
        #         "aggregate_height_penalty: ", aggregate_height_penalty,
        #         "hole_penalty: ", hole_penalty,
        #         "bumpiness_penalty: ", bumpiness_penalty,
        #         "survival_bonus: ", survival_bonus,
        #         "lost_penalty: ", lost_penalty)

        return value

    def calculate_height_penalty(self, grid):
        """Calculate penalty based on column heights"""
        heights = []
        for col in range(10):
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    heights.append(20 - row)
                    break
            else:
                heights.append(0)
        return max(heights)

    def count_potential_holes(self, grid):
        """Count potential holes that could be created"""
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block:
                    if row < 19 and grid[row + 1][col] != (0, 0, 0):
                        holes += 1
        return holes

    def count_empty_columns(self, grid):
        """Count the number of empty columns in the grid"""
        empty_columns = 0
        for col in range(10):
            if all(grid[row][col] == (0, 0, 0) for row in range(20)):
                empty_columns += 1
        return empty_columns

def make_tetris_env(window_id=None):
    def _init():
        return GymTetrisEnv(window_id=window_id)
    return _init


class GymTetrisEnv(gym.Env):
    def __init__(self, window_id=None):
        super().__init__()
        self.env = TetrisEnv(window_id=window_id)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(200 + 7 + 3 + 7,),  # total = 217
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(5)



    def close(self):
        self.env.close()

    def step(self, action):
        _, reward, done = self.env.step(action)
        obs = self.env.get_state()
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset_game()
        info = {}
        return obs, info

    def preprocess_env(self, grid):
        return np.array([[1 if cell != (0, 0, 0) else 0 for cell in row] for row in grid], dtype=np.float32)


def get_piece_height(piece):
    positions = convert_shape_format(piece)
    if not positions:
        return 0
    y_coords = [pos[1] for pos in positions]
    return max(y_coords)


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
        x, y = pos
        if x < 0 or x >= 10 or y >= 20:
            return False
        if pos not in accepted_positions and y >= 0:
            return False
        if y >= 0 and grid[y][x] != (0, 0, 0):
            return False

    return True




def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False

def get_valid_final_positions(grid, piece):
    placements = []
    for rotation in range(len(piece.shape)):
        test_piece = Piece(0, 0, piece.shape)
        test_piece.rotation = rotation

        for x in range(10):
            test_piece.x = x
            test_piece.y = 0

            # Drop piece down
            while valid_space(test_piece, grid):
                test_piece.y += 1
            test_piece.y -= 1

            if valid_space(test_piece, grid):
                placements.append((x, rotation, test_piece.y))
    return placements

def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont('Arial', size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width() / 2), top_left_y + play_height/2 - (label.get_height() / 2)))


def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y
    for i in range(len(grid)):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i*block_size), (sx + play_width, sy + i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * block_size, sy), (sx + j * block_size, sy + play_height))

def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except ValueError:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                new_key = (x, y + inc)
                locked[new_key] = locked.pop(key)
    return inc


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('Arial', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = top_left_x + play_width + 50 #TODO: Change these constants as wanted
    sy = top_left_y + play_height / 2 - 100

    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*block_size, sy + i*block_size, block_size, block_size), 0)

    surface.blit(label, (sx + 10, sy - 30))


def draw_window(surface, grid, score=0, high_score=0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('Arial', 60)  # TODO: change font and size
    label = font.render('Tetris', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))


    #Current Score
    font = pygame.font.SysFont('Arial', 30)
    label = font.render(('Score: ' + str(score)), 1, (255, 255, 255))

    sx = top_left_x + play_width + 50  # TODO: Change these constants as wanted
    sy = top_left_y + play_height / 2 - 100

    surface.blit(label, (sx + 20, sy + 160))

    #High Score
    label = font.render(('High Score: ' + str(high_score)), 1, (255, 255, 255))

    sx = top_left_x - 240  # TODO: Change these constants as wanted
    sy = top_left_y + 200

    surface.blit(label, (sx + 20, sy + 160))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)


    draw_grid(surface, grid)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 4)

    # pygame.display.update()

def count_holes(grid):
    holes = 0
    for col in range(len(grid[0])):
        block_seen = False
        for row in range(len(grid)):
            if grid[row][col] != (0, 0, 0):
                block_seen = True
            elif block_seen and grid[row][col] == (0, 0, 0):
                holes += 1
    # print(holes)
    return holes



def main_menu():
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import RecordEpisodeStatistics

    env = DummyVecEnv([make_tetris_env()])
    # env = RecordEpisodeStatistics(env)

    # if os.path.exists("tetris_dqn_model.zip"):
    if False:
        print("Loading saved model...")
        model = DQN.load("tetris_dqn_model.zip", env=env)
    else:
        print("Training new model...")
        model = DQN("MlpPolicy", env, verbose=1, buffer_size=100000)

    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("tetris_dqn_model")

    # Run the trained model
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    global HIGH_SCORE
    if model.get_env().envs[0].env.score > HIGH_SCORE:
        with open('scores.txt', 'w') as f:
            f.write(str(model.get_env().envs[0].env.score))

    model.save("tetris_dqn_model")
    model.save_replay_buffer("tetris_dqn_buffer")

def rerun_model():
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    import os

    env = DummyVecEnv([make_tetris_env()])

    # Load model
    model = DQN.load("tetris_dqn_model.zip", env=env)

    # if os.path.exists("tetris_dqn_buffer.pkl"):
    #     model.load_replay_buffer("tetris_dqn_buffer")


    obs = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # done = terminated or truncated

        if done:
            obs = env.reset()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from multiprocessing import freeze_support
    freeze_support()
    main_menu()
    # rerun_model()

