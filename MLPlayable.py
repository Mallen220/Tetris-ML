import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

from pygame import surface
from pygame.examples import grid
from stable_baselines3.common.monitor import Monitor

import logging
import csv
from datetime import datetime

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
episode_counter = 0

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

# shapes = [I]
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

# Setup logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tetris_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


csv_file = os.path.join(log_dir, f"tetris_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_headers = ["episode", "Reward", "Cleared", "setup_reward", "flatness_bonus", "progress_reward",
               "survival_bonus", "score_reward", "height_penalty", "aggregate_height_penalty", "holes", "well_penalty", "lost_penalty"]

csv_fp = open(csv_file, mode='w', newline='')
csv_writer = csv.DictWriter(csv_fp, fieldnames=csv_headers)
csv_writer.writeheader()

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console.setFormatter(formatter)
# logger.addHandler(console)



class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(self.shape)]
        self.rotation = 0

def create_grid(locked_positions={}):
    grid = [[(0, 0,0 ) for x in range(10)] for y in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                color = locked_positions[(j, i)]
                grid[i][j] = color

    return grid

class TetrisEnv:
    def __init__(self, window_id=None, render_enabled=True):
        self.render_enabled = render_enabled  # toggle this to False if running headless
        if self.render_enabled:
            pygame.init()

        self.window_id = window_id
        if self.render_enabled:
            self.win = pygame.display.set_mode((s_width, s_height))
        else:
            self.win = None

        self.possible_next_pieces = shapes.copy()
        self.run = True
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.27  # controls how fast the pieces fall
        self.level_time = 0
        self.score = 0
        self.change_piece = False
        self.last_aggregate_height = 0
        self.last_bumpiness = 0
        self.last_holes = 0

        if self.render_enabled:
            self.window = pygame.display.set_mode((s_width, s_height))
            pygame.display.set_caption('Tetris AI')
        else:
            self.window = None

        self.clock = pygame.time.Clock()
        self.fast_mode = False
        self.done = check_lost(self.locked_positions)
        self.start_time = pygame.time.get_ticks()

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

    def get_shape(self):
        if len(self.possible_next_pieces) == 0:
            self.possible_next_pieces = shapes.copy()

        new_piece = random.choice(self.possible_next_pieces)
        self.possible_next_pieces.remove(new_piece)
        return Piece(5, 0, new_piece)

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
        self.next_piece = self.get_shape()
        self.grid = create_grid(self.locked_positions)
        cleared = clear_rows(self.grid, self.locked_positions)
        self.score += cleared * 10
        self.done = check_lost(self.locked_positions)

    def step(self, action):
        # print(action)

        # self.fall_time += pygame.time.get_ticks()
        # self.level_time += pygame.time.get_ticks()
        self.render()

        if action == 4:
            if not hasattr(self, "pending_actions") or not self.pending_actions:
                plan = self.get_best_plan()
                self.pending_actions = self.get_action_sequence(plan)

            next_action = self.pending_actions.pop(0)
            if next_action == "drop":
                while valid_space(self.current_piece, self.grid):
                    self.current_piece.y += 1
                self.current_piece.y -= 1
            else:
                self.do_action(next_action)
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
            self.next_piece = self.get_shape()

        # Update grid and score
        self.grid = create_grid(self.locked_positions)

        ### REWARD MATH
        self.done = check_lost(self.locked_positions)

        cleared = clear_rows(self.grid, self.locked_positions)
        self.score += cleared * 10

        reward = self.evaluate_board(self.grid, cleared, True)

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

    def get_best_plan(self):
        best_value = float('-inf')
        best_plan = None

        for rotation in range(len(self.current_piece.shape)):
            for x in range(-2, 12):
                test_piece = Piece(self.current_piece.x, self.current_piece.y, self.current_piece.shape)
                test_piece.rotation = rotation
                test_piece.x = x

                while valid_space(test_piece, self.grid):
                    test_piece.y += 1
                test_piece.y -= 1

                if not valid_space(test_piece, self.grid):
                    continue

                locked_copy = self.locked_positions.copy()
                for pos in convert_shape_format(test_piece):
                    if pos[1] > -1:
                        locked_copy[(pos[0], pos[1])] = test_piece.color

                grid_copy = create_grid(locked_copy)
                cleared = clear_rows(grid_copy, locked_copy)

                value = self.evaluate_board(grid_copy, cleared)

                if value > best_value:
                    best_value = value
                    best_plan = {
                        "target_x": x,
                        "target_rotation": rotation,
                        "drop_y": test_piece.y
                    }

        return best_plan

    def get_action_sequence(self, plan): #Converts a plan into a sequence of actions
        actions = []
        # Rotation first
        current_rot = self.current_piece.rotation % len(self.current_piece.shape)
        while current_rot != plan["target_rotation"]:
            actions.append(3)  # rotate
            current_rot = (current_rot + 1) % len(self.current_piece.shape)

        # Horizontal movement
        curr_x = self.current_piece.x
        while curr_x < plan["target_x"]:
            actions.append(1)  # move right
            curr_x += 1  # simulate
        while curr_x > plan["target_x"]:
            actions.append(0)  # move left
            curr_x -= 1  # simulate

        # Drop to final Y
        actions.append("drop")

        return actions

    def evaluate_board(self, grid, cleared=0, logs=False):
        """Evaluate the quality of a board position""" #TODO modify this function to include more heuristics
        """Returns Reward Value"""
        # Calculate height
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

        # Include walls in bumpiness to avoid edge bias
        if len(heights) > 0:
            bumpiness += heights[0] # Left wall (height 0)
            bumpiness += heights[-1] # Right wall (height 0)

        # Count holes
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if grid[row][col] != (0, 0, 0):
                    found_block = True
                elif found_block and grid[row][col] == (0, 0, 0):
                    holes += 1

        # 1. Reward for potential future clears (encourage setups)
        setup_reward = 0
        # for col in range(10):
        #     # Reward columns with 1-2 blocks at the top of stacks
        #     for row in range(19, 0, -1):
        #         if grid[row][col] == (0, 0, 0) and grid[row - 1][col] != (0, 0, 0):
        #             # Higher reward for gaps near the top
        #             setup_reward += (20 - row) * 0.5
        #             break
        # setup_reward *= 0.05  # Scale down the setup reward

        # 2. Flatness bonus (penalize uneven surfaces)
        flatness_bonus = 0
        if hasattr(self, 'last_bumpiness'):
            flatness_bonus = (self.last_bumpiness - bumpiness) * 0.5

        if logs:
            self.last_bumpiness = bumpiness

        # 3. Deep well detection (prevent stuck pieces)
        well_penalty = 0

        # 4. Survival bonus (scaled by time survived)
        survival_bonus = 1

        # 5. Clear rewards with Tetris emphasis
        clear_reward = cleared * 100
        if cleared >= 4:
            clear_reward += 500  # Big Tetris bonus
        elif cleared > 0:
            clear_reward += 20 * (4 - cleared)  # Bonus for partial setups

        # 6. Height penalties (more aggressive)
        # height_penalty = max_height * -2 # Removed absolute penalty
        # aggregate_height_penalty = aggregate_height * -0.5 # Removed absolute penalty

        # 7. Hole penalties (critical!)
        # hole_penalty = holes * -10  # Removed absolute penalty
        hole_penalty = 0
        if hasattr(self, 'last_holes'):
            hole_penalty = (self.last_holes - holes) * 10 # Reward for reducing holes, penalty for adding them

        if logs:
            self.last_holes = holes

        # Calculate how much the board has changed
        progress_reward = 0
        if hasattr(self, 'last_aggregate_height'):
            height_diff = aggregate_height - self.last_aggregate_height
            # Reward for reducing height (clearing lines)
            progress_reward = -height_diff * 0.5 # Penalty for increasing height, reward for decreasing

        if logs:
            self.last_aggregate_height = aggregate_height

        # score_reward = self.score * 1
        # print(score_reward)

        lost_penalty = (-50 if self.done else 0)  # Game over penalty
        # Calculate total value
        value = (
                clear_reward +
                # setup_reward +
                flatness_bonus +
                progress_reward +
                # score_reward
                survival_bonus +
                # height_penalty +
                # aggregate_height_penalty +
                hole_penalty +
                # well_penalty +
                lost_penalty
        )

        # value = 1 + (cleared ** 2) * 10  # BOARD_WIDTH = 10
        # if self.done:
        #     value -= 2

        if logs:
            global episode_counter
            if self.done:
                episode_counter += 1

            logger.info(f"Reward: {round(value, 2)}, "
                        f"Cleared: {round(cleared, 2)}, "
                        # f"setup_reward: {round(setup_reward, 2)}, "
                        f"flatness_bonus: {round(flatness_bonus, 2)}, "
                        f"progress_reward: {round(progress_reward, 2)}, "
                        f"survival_bonus: {round(survival_bonus, 2)}, "
                        # f"height_penalty: {round(height_penalty, 2)}, "
                        # f"aggregate_height_penalty: {round(aggregate_height_penalty, 2)}, "
                        f"holes: {round(hole_penalty, 2)}, "
                        # f"well_penalty: {round(well_penalty, 2)}, "
                        f"lost_penalty: {round(lost_penalty, 2)}"
            )
            csv_writer.writerow({
                "episode": episode_counter,
                "Reward": round(value, 2),
                "Cleared": round(cleared, 2),
                # "setup_reward": round(setup_reward, 2),
                "flatness_bonus": round(flatness_bonus, 2),
                "progress_reward": round(progress_reward, 2),
                "survival_bonus": round(survival_bonus, 2),
                # "height_penalty": round(height_penalty, 2),
                # "aggregate_height_penalty": round(aggregate_height_penalty, 2),
                "holes": round(hole_penalty, 2),
                # "well_penalty": round(well_penalty, 2),
                "lost_penalty": round(lost_penalty, 2)
            })
            csv_fp.flush()

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

def make_tetris_env(window_id=None, render_enabled=True):
    def _init():
        return GymTetrisEnv(window_id=window_id, render_enabled=render_enabled)
    return _init


class GymTetrisEnv(gym.Env):
    def __init__(self, window_id=None, render_enabled=True):
        super().__init__()
        self.env = TetrisEnv(window_id=window_id, render_enabled=render_enabled)

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
    rows_to_clear = []
    for i in range(len(grid)-1, -1, -1):
        if (0, 0, 0) not in grid[i]:
            rows_to_clear.append(i)
            for j in range(len(grid[i])):
                try:
                    del locked[(j, i)]
                except KeyError:
                    pass
    if rows_to_clear:
        # Sort in ascending order to shift properly
        rows_to_clear.sort()
        for row in rows_to_clear:
            # Move all rows above this row down by 1
            for key in sorted(locked.copy(), key=lambda x: x[1]):
                x, y = key
                if y < row:
                    new_key = (x, y + 1)
                    locked[new_key] = locked.pop(key)

    return len(rows_to_clear)



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
    import time

    global csv_writer, csv_fp


    env = DummyVecEnv([make_tetris_env(render_enabled=False)])
    # env = RecordEpisodeStatistics(env)

    if os.path.exists("tetris_dqn_model.zip"):
    # if False:
        print("Loading saved model...")
        model = DQN.load("tetris_dqn_model.zip", env=env)
    else:
        print("Training new model...")
        # model = DQN("MlpPolicy", env, verbose=1, buffer_size=100000)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=200000,  # Larger buffer for complex game
            learning_starts=10000,  # Collect experiences before learning
            target_update_interval=1000,  # Update target network
            train_freq=4,  # Update every 4 steps
            gradient_steps=1,  # How many gradient steps per update
            exploration_fraction=0.2,  # Longer exploration
            exploration_final_eps=0.02  # Lower final exploration
        )



    model.learn(total_timesteps=200000, progress_bar=True)
    model.save("tetris_dqn_model")

    # Run the trained model
    obs = env.reset()
    done = False


    while not done:

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
    # obs = env.reset()

    global HIGH_SCORE
    if model.get_env().envs[0].env.score > HIGH_SCORE:
        with open('scores.txt', 'w') as f:
            f.write(str(model.get_env().envs[0].env.score))

    model.save("tetris_dqn_model")
    model.save_replay_buffer("tetris_dqn_buffer")
    csv_fp.close()
    logger.info("Finished all episodes. Model saved. CSV file closed.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from multiprocessing import freeze_support
    freeze_support()
    main_menu()
    # rerun_model()