import pygame
import pandas as pd
import numpy as np
import os
import time
import threading
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime


class TetrisVisualizer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.DataFrame()
        self.last_update = time.time()
        self.last_file_size = 0
        self.update_interval = 1  # seconds
        self.current_episode_range = (0, 100)
        self.dragging = False
        self.drag_start = 0
        self.selected_metrics = []
        self.user_set_range = False  # Track if user has manually set the range
        self.available_metrics = []  # Will be populated from CSV headers

        # Initialize pygame
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tetris Training Dashboard")

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasAgg(self.fig)

        # Load initial data
        self.load_data()
        self.running = True
        self.render_visualization()

    def load_data(self):
        if os.path.exists(self.csv_path):
            try:
                # Check if file has actually changed
                current_size = os.path.getsize(self.csv_path)
                if current_size == self.last_file_size:
                    return False  # No new data

                new_data = pd.read_csv(self.csv_path)
                if not new_data.empty:
                    # Store file size for future comparisons
                    self.last_file_size = current_size

                    # Update available metrics based on CSV headers
                    self.update_available_metrics(new_data)

                    # Preserve user's metric selections if possible
                    self.preserve_selected_metrics()

                    # Only update range if user hasn't set it manually
                    if not self.user_set_range:
                        max_ep = new_data['episode'].max()
                        self.current_episode_range = (max(0, max_ep - 50), max_ep)

                    self.data = new_data
                    return True  # New data loaded
            except Exception as e:
                print(f"Error loading data: {e}")
        return False  # No new data loaded

    def update_available_metrics(self, new_data):
        """Extract available metrics from CSV headers"""
        # Exclude non-metric columns
        non_metric_cols = ['episode', 'step', 'timestamp', 'datetime', 'date']
        self.available_metrics = [
            col for col in new_data.columns
            if col not in non_metric_cols and
               pd.api.types.is_numeric_dtype(new_data[col])
        ]

        # Set initial selections if none exist
        if not self.selected_metrics and self.available_metrics:
            # Try to select common metrics, or first 4 if not found
            preferred = ['Reward', 'Cleared', 'setup_reward', 'flatness_bonus']
            self.selected_metrics = [m for m in preferred if m in self.available_metrics]
            if not self.selected_metrics:
                self.selected_metrics = self.available_metrics[:4]

    def preserve_selected_metrics(self):
        """Maintain user selections where possible"""
        # Filter out metrics that are no longer available
        self.selected_metrics = [
            m for m in self.selected_metrics
            if m in self.available_metrics
        ]

        # If all selections were removed, select first available
        if not self.selected_metrics and self.available_metrics:
            self.selected_metrics = self.available_metrics[:4]

    def run(self):
        while self.running:
            current_time = time.time()
            if current_time - self.last_update > self.update_interval:
                data_loaded = self.load_data()
                if data_loaded:
                    self.render_visualization()
                    self.last_update = current_time

            self.handle_events()
            time.sleep(0.01)

        pygame.quit()

    def render_visualization(self):
        # Clear screen
        self.screen.fill((30, 30, 40))

        if self.data.empty or 'episode' not in self.data.columns:
            font = pygame.font.SysFont(None, 48)
            text = font.render("Waiting for data...", True, (200, 200, 200))
            self.screen.blit(text, (self.width // 2 - text.get_width() // 2,
                                    self.height // 2 - text.get_height() // 2))
            pygame.display.flip()
            return

        # Get visible episode range
        start_ep, end_ep = self.current_episode_range
        visible_data = self.data[(self.data['episode'] >= start_ep) &
                                 (self.data['episode'] <= end_ep)]

        # Plot selected metrics
        self.ax.clear()
        for metric in self.selected_metrics:
            if metric in visible_data.columns:
                # Aggregate data by episode
                episode_data = visible_data.groupby('episode')[metric].mean().reset_index()
                self.ax.plot(episode_data['episode'], episode_data[metric], label=metric, linewidth=2)

        # Format plot
        self.ax.set_title(f"Training Metrics (Episodes {start_ep}-{end_ep})", fontsize=14, color='white')
        self.ax.set_xlabel("Episode", color='white')
        self.ax.set_ylabel("Metric Value", color='white')

        # Only show legend if we have metrics to display
        if self.selected_metrics:
            self.ax.legend(loc='upper right')

        self.ax.grid(True, alpha=0.2)
        self.ax.set_facecolor((0.1, 0.1, 0.15))
        self.fig.patch.set_facecolor((0.1, 0.1, 0.15))
        self.ax.tick_params(colors='white')

        # Set plot colors
        for spine in self.ax.spines.values():
            spine.set_color('gray')

        # Render matplotlib figure to buffer
        self.canvas.draw()

        # Get the image as an RGB array
        buf = self.canvas.buffer_rgba()
        w, h = self.canvas.get_width_height()

        # Convert to pygame surface (using RGBA to preserve transparency)
        img = pygame.image.frombuffer(buf, (w, h), "RGBA")

        # Scale to fit available space
        target_width = min(w, self.width - 100)
        target_height = min(h, self.height - 200)
        img = pygame.transform.smoothscale(img, (target_width, target_height))

        # Position image
        img_x = (self.width - target_width) // 2
        img_y = 50
        self.screen.blit(img, (img_x, img_y))

        # Draw UI controls
        self.draw_controls()

        # Update display
        pygame.display.flip()

    def draw_controls(self):
        # Draw timeline slider
        pygame.draw.rect(self.screen, (60, 60, 80), (50, 650, 1100, 30))
        max_ep = self.data['episode'].max() if not self.data.empty else 100

        # Calculate slider positions
        start_x = 50 + 1100 * (self.current_episode_range[0] / max_ep) if max_ep > 0 else 50
        end_x = 50 + 1100 * (self.current_episode_range[1] / max_ep) if max_ep > 0 else 1150

        # Draw selection range
        pygame.draw.rect(self.screen, (0, 150, 200), (start_x, 650, end_x - start_x, 30))

        # Draw handles
        pygame.draw.circle(self.screen, (255, 255, 255), (int(start_x), 665), 12)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(end_x), 665), 12)

        # Draw metric selection buttons
        font = pygame.font.SysFont(None, 24)
        button_width = 100
        max_buttons = (self.width - 100) // button_width

        for i, metric in enumerate(self.available_metrics[:max_buttons]):
            color = (100, 200, 100) if metric in self.selected_metrics else (100, 100, 100)
            pygame.draw.rect(self.screen, color, (50 + i * button_width, 700, button_width - 5, 40))

            # Truncate long metric names
            display_text = metric[:10] + "..." if len(metric) > 10 else metric
            text = font.render(display_text, True, (255, 255, 255))
            self.screen.blit(text, (55 + i * button_width, 710))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                # Check if dragging timeline
                if 650 <= y <= 680:
                    self.dragging = True
                    self.drag_start = x
                    self.user_set_range = True  # User is manually setting range

                # Check metric selection buttons
                if 700 <= y <= 740:
                    button_width = 100
                    max_buttons = (self.width - 100) // button_width
                    available = self.available_metrics[:max_buttons]

                    for i, metric in enumerate(available):
                        if 50 + i * button_width <= x <= 50 + (i + 1) * button_width:
                            if metric in self.selected_metrics:
                                self.selected_metrics.remove(metric)
                            else:
                                self.selected_metrics.append(metric)
                            self.render_visualization()

            if event.type == pygame.MOUSEBUTTONUP:
                self.dragging = False

            if event.type == pygame.MOUSEMOTION and self.dragging:
                dx = event.pos[0] - self.drag_start
                self.drag_start = event.pos[0]
                self.user_set_range = True  # User is manually setting range

                max_ep = self.data['episode'].max() if not self.data.empty else 100
                range_size = self.current_episode_range[1] - self.current_episode_range[0]

                # Calculate new range
                new_start = max(0, self.current_episode_range[0] + (dx / 1100) * max_ep)
                new_end = min(max_ep, new_start + range_size)

                # Prevent inversion
                if new_end > new_start:
                    self.current_episode_range = (new_start, new_end)
                    self.render_visualization()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in
                    current_range = self.current_episode_range[1] - self.current_episode_range[0]
                    new_range = max(10, current_range * 0.8)
                    self.adjust_range(new_range)
                    self.user_set_range = True  # User is manually setting range

                if event.key == pygame.K_MINUS:
                    # Zoom out
                    current_range = self.current_episode_range[1] - self.current_episode_range[0]
                    max_ep = self.data['episode'].max() if not self.data.empty else 100
                    new_range = min(max_ep, current_range * 1.2)
                    self.adjust_range(new_range)
                    self.user_set_range = True  # User is manually setting range

        return True

    def adjust_range(self, new_range):
        center = sum(self.current_episode_range) / 2
        new_start = max(0, center - new_range / 2)
        new_end = min(self.data['episode'].max() if not self.data.empty else 100,
                      center + new_range / 2)
        self.current_episode_range = (new_start, new_end)
        self.render_visualization()


def run_visualizer(csv_file):
    visualizer = TetrisVisualizer(csv_file)
    visualizer.run()


if __name__ == "__main__":
    import sys
    import glob
    import os


    # Get the newest file in logs directory
    def get_newest_csv():
        log_dir = "logs"
        if not os.path.exists(log_dir):
            print(f"Error: Directory '{log_dir}' does not exist")
            return None

        csv_files = glob.glob(os.path.join(log_dir, "tetris_metrics_*.csv"))
        if not csv_files:
            print(f"Error: No tetris_metrics_*.csv files found in '{log_dir}'")
            return None

        # Sort by modification time (newest first)
        csv_files.sort(key=os.path.getmtime, reverse=True)
        return csv_files[0]


    # Try to get the newest file
    newest_file = get_newest_csv()
    if newest_file:
        print(f"Using newest metrics file: {newest_file}")
        run_visualizer(newest_file)
    else:
        print("No valid metrics file found. Exiting.")
        sys.exit(1)