
import tkinter as tk
import numpy as np
import random
import time
import threading
from tkinter import messagebox

GRID_ROWS = 6
GRID_COLS = 6
CELL_SIZE = 50
START_POS = (0, 0)
GOAL_POS = (5, 5)

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Builder - Click to Add/Remove Walls")
        self.canvas = tk.Canvas(self.root, width=GRID_COLS * CELL_SIZE, height=GRID_ROWS * CELL_SIZE)
        self.canvas.pack()
        self.grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)  # 0 = empty, 1 = wall
        self.rectangles = {}
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.episodes = 200
        self.best_steps = float("inf")
        self.best_path = []

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_wall)

        self.run_button = tk.Button(self.root, text="Run SARSA", command=self.run_sarsa_thread)
        self.run_button.pack(pady=10)

    def draw_grid(self):
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
                self.rectangles[(i, j)] = rect
        self.update_cell(START_POS, "green")
        self.update_cell(GOAL_POS, "red")

    def toggle_wall(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if (row, col) == START_POS or (row, col) == GOAL_POS:
            return
        self.grid[row][col] = 1 - self.grid[row][col]
        new_color = "black" if self.grid[row][col] == 1 else "white"
        self.update_cell((row, col), new_color)

    def update_cell(self, pos, color):
        self.canvas.itemconfig(self.rectangles[pos], fill=color)

    def get_possible_actions(self, state):
        actions = []
        directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        for action, (dr, dc) in directions.items():
            r, c = state[0] + dr, state[1] + dc
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and self.grid[r][c] == 0:
                actions.append(action)
        return actions

    def take_action(self, state, action):
        directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = directions[action]
        return (state[0] + dr, state[1] + dc)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            actions = self.get_possible_actions(state)
            return random.choice(actions) if actions else None
        else:
            self.q_table.setdefault(state, {})
            state_actions = self.q_table[state]
            if not state_actions:
                return random.choice(self.get_possible_actions(state))
            return max(state_actions, key=state_actions.get)

    def run_sarsa(self):
        for ep in range(self.episodes):
            state = START_POS
            action = self.choose_action(state)
            steps = 0
            path = [state]

            while state != GOAL_POS and action:
                next_state = self.take_action(state, action)
                reward = 1 if next_state == GOAL_POS else -0.1
                next_action = self.choose_action(next_state)

                self.q_table.setdefault(state, {})
                self.q_table[state].setdefault(action, 0)
                self.q_table.setdefault(next_state, {})
                self.q_table[next_state].setdefault(next_action, 0)

                td_target = reward + self.gamma * self.q_table[next_state][next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error

                self.update_cell(state, "lightblue")
                self.root.update()
                time.sleep(0.01)

                state, action = next_state, next_action
                path.append(state)
                steps += 1

            if state == GOAL_POS and steps < self.best_steps:
                self.best_steps = steps
                self.best_path = path.copy()

            print(f"Episode {ep+1}: Steps taken = {steps}")
            self.reset_visuals()

        self.show_best_path()

    def reset_visuals(self):
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                if (i, j) != START_POS and (i, j) != GOAL_POS and self.grid[i][j] == 0:
                    self.update_cell((i, j), "white")
        self.update_cell(START_POS, "green")
        self.update_cell(GOAL_POS, "red")

    def show_best_path(self):
        for pos in self.best_path:
            if pos != START_POS and pos != GOAL_POS:
                self.update_cell(pos, "yellow")
                self.root.update()
                time.sleep(0.05)
        messagebox.showinfo("Training Complete", f"Shortest path found in {self.best_steps} steps!")

    def run_sarsa_thread(self):
        threading.Thread(target=self.run_sarsa).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()
