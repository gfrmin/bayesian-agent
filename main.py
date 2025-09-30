#!/usr/bin/env python3
"""Main simulation loop with terminal rendering."""

import curses
import time
import sys
from typing import Optional
from environment import GridWorld
from bayesian_agent import BayesianAgent
from config import (
    GRID_WIDTH, GRID_HEIGHT, UPDATE_DELAY, SHOW_BELIEFS,
    COLOR_CODES, SHAPES, COLORS
)


class Simulation:
    """Main simulation with rendering."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.env = GridWorld()
        self.agent = BayesianAgent()
        self.running = True
        self.paused = False
        self.current_target: Optional[tuple] = None

        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100)  # 100ms timeout for getch()

        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        for color_name, color_id in COLOR_CODES.items():
            curses.init_pair(color_id, self._get_curses_color(color_name), -1)

        # Stats
        self.start_time = time.time()

    def _get_curses_color(self, color_name: str) -> int:
        """Map color name to curses color constant."""
        color_map = {
            "red": curses.COLOR_RED,
            "green": curses.COLOR_GREEN,
            "yellow": curses.COLOR_YELLOW,
            "blue": curses.COLOR_BLUE,
        }
        return color_map.get(color_name, curses.COLOR_WHITE)

    def render(self):
        """Render the current state."""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # Check if terminal is large enough
        min_height = 24
        min_width = 80
        if height < min_height or width < min_width:
            self.stdscr.addstr(0, 0, f"Terminal too small! Need at least {min_width}x{min_height}")
            self.stdscr.refresh()
            return

        # Draw grid border
        for y in range(GRID_HEIGHT + 2):
            self.stdscr.addstr(y, 0, "|")
            self.stdscr.addstr(y, GRID_WIDTH + 1, "|")
        self.stdscr.addstr(0, 0, "+" + "-" * GRID_WIDTH + "+")
        self.stdscr.addstr(GRID_HEIGHT + 1, 0, "+" + "-" * GRID_WIDTH + "+")

        # Draw food items
        for food in self.env.foods:
            x, y = food.position
            color_pair = COLOR_CODES[food.color]
            try:
                self.stdscr.addstr(y + 1, x + 1, food.shape, curses.color_pair(color_pair))
            except curses.error:
                pass  # Ignore if position is out of bounds

        # Draw agent
        ax, ay = self.agent.position
        try:
            self.stdscr.addstr(ay + 1, ax + 1, "@", curses.A_BOLD)
        except curses.error:
            pass

        # Draw target path if exists
        if self.current_target:
            tx, ty = self.current_target
            try:
                self.stdscr.addstr(ty + 1, tx + 1, "X", curses.A_DIM)
            except curses.error:
                pass

        # Draw compact stats below grid
        info_y = GRID_HEIGHT + 2

        # Agent stats on one line
        state = self.agent.get_state()
        elapsed = time.time() - self.start_time
        status = "PAUSED" if self.paused else f"Run:{elapsed:.0f}s"
        info_line = f"{status} | E:{self.agent.energy:.1f} | Pos:{self.agent.position} | Steps:{self.agent.total_steps} | Obs:{len(self.agent.observations)} | Foods:{len(self.env.foods)}"
        try:
            self.stdscr.addstr(info_y, 0, info_line[:width-1])
        except curses.error:
            pass

        # Belief summary (compact)
        if SHOW_BELIEFS and info_y + 2 < height:
            beliefs = self.agent.get_belief_summary()

            # Show beliefs in rows by shape
            belief_y = info_y + 1
            for shape in SHAPES:
                if belief_y >= height - 1:
                    break

                belief_line = f"{shape} "
                for color in COLORS:
                    key = (shape, color)
                    belief = beliefs[key]
                    mean = belief["mean"]
                    std = belief["std"]
                    belief_line += f"{color[0].upper()}:{mean:+.1f} "

                try:
                    self.stdscr.addstr(belief_y, 0, belief_line[:width-1])
                except curses.error:
                    pass
                belief_y += 1

        # Controls at bottom
        if height > info_y + 5:
            try:
                self.stdscr.addstr(height - 2, 0, "SPACE: Pause | q: Quit")
            except curses.error:
                pass

        self.stdscr.refresh()

    def handle_input(self):
        """Handle keyboard input."""
        try:
            key = self.stdscr.getch()
            if key == ord('q'):
                self.running = False
            elif key == ord(' '):
                self.paused = not self.paused
        except:
            pass

    def update(self):
        """Update simulation state."""
        if self.paused:
            return

        # Update environment (spawn food)
        self.env.update()

        # Agent decision making
        # Get visible foods
        nearby_foods = self.env.get_nearby_foods(self.agent.position, radius=50)
        available_foods = [(f.position, f.shape, f.color) for f in nearby_foods]

        # Check if agent is at current target
        if self.current_target == self.agent.position:
            self.current_target = None

        # Select new target if needed
        if self.current_target is None:
            self.current_target = self.agent.select_target_food(available_foods)

        # Move toward target
        if self.current_target:
            # Check if we're at target
            if self.agent.position == self.current_target:
                # Try to consume food
                observation = self.env.consume_food(self.agent.position)
                if observation:
                    shape, color, energy = observation
                    self.agent.consume(shape, color, energy)
                self.current_target = None
            else:
                # Move toward target
                next_pos = self.agent.get_next_move(self.current_target)
                self.agent.move(next_pos)

                # Check if we reached food at new position
                observation = self.env.consume_food(self.agent.position)
                if observation:
                    shape, color, energy = observation
                    self.agent.consume(shape, color, energy)
                    self.current_target = None

        # Check if agent is out of energy
        if self.agent.energy <= 0:
            self.running = False

    def run(self):
        """Main simulation loop."""
        # Spawn initial food
        for _ in range(5):
            self.env.spawn_food()

        while self.running:
            self.handle_input()
            self.update()
            self.render()
            time.sleep(UPDATE_DELAY)

        # Show final stats
        self.show_final_stats()

    def show_final_stats(self):
        """Display final statistics."""
        # Print to stdout for scrollback (curses.wrapper will handle endwin)
        print("\n=== SIMULATION ENDED ===")
        print(f"Final Energy: {self.agent.energy:.2f} | Steps: {self.agent.total_steps} | Foods Consumed: {len(self.env.eaten_foods)}")

        # Show learned beliefs
        print("\n=== LEARNED BELIEFS ===")
        beliefs = self.agent.get_belief_summary()

        from config import ENERGY_DISTRIBUTIONS
        for key in sorted(beliefs.keys()):
            shape, color = key
            belief = beliefs[key]
            true_mean, true_std = ENERGY_DISTRIBUTIONS[key]

            print(f"{shape} {color:6s}: learned={belief['mean']:+.1f}±{belief['std']:.1f} true={true_mean:+.1f}±{true_std:.1f} (n={int(belief['n'])})")

        # Show energy history
        print("\n=== ENERGY HISTORY ===")
        if self.agent.energy_history:
            for idx, entry in enumerate(self.agent.energy_history, 1):
                before = entry["energy_before"]
                after = entry["energy_after"]
                delta = entry["delta"]
                reason = entry["reason"]

                print(f"Step {idx:3d}: {before:6.2f}→{after:6.2f} (Δ{delta:+.2f}) {reason}")
        else:
            print("No energy history recorded.")

        print()


def main(stdscr):
    """Entry point for curses."""
    sim = Simulation(stdscr)
    sim.run()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
        sys.exit(0)