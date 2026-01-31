"""Curses-based visualization of population foraging on grid."""

import curses
import time
import random
from simulation import Simulation, SimulationConfig
from reality import FoodAttribute


SHAPE_GLYPHS = {
    'circle': '\u25cf',
    'square': '\u25a0',
    'triangle': '\u25b2',
    'hexagon': '\u2b21',
}

COLOR_PAIRS = {
    'red': 1,
    'green': 2,
    'yellow': 3,
    'blue': 4,
}

AGENT_COLOR_PAIR = 5


class PopulationVisualizer:
    """Real-time curses visualization of a population foraging simulation."""

    def __init__(self, stdscr, simulation: Simulation):
        self.stdscr = stdscr
        self.sim = simulation
        self.paused = False
        self.delay = 0.05
        self.tick_in_gen = 0

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(COLOR_PAIRS['red'], curses.COLOR_RED, -1)
        curses.init_pair(COLOR_PAIRS['green'], curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_PAIRS['yellow'], curses.COLOR_YELLOW, -1)
        curses.init_pair(COLOR_PAIRS['blue'], curses.COLOR_BLUE, -1)
        curses.init_pair(AGENT_COLOR_PAIR, curses.COLOR_CYAN, -1)

    def render(self) -> None:
        """Draw the current simulation state."""
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        gw, gh = self.sim.config.grid_size

        # Grid border
        border_w = gw + 2
        border_h = gh + 2
        if height < border_h + 6 or width < border_w + 2:
            self._safe_addstr(0, 0, f"Terminal too small ({width}x{height}), need {border_w+2}x{border_h+6}")
            self.stdscr.refresh()
            return

        # Top/bottom borders
        self._safe_addstr(0, 0, '+' + '-' * gw + '+')
        self._safe_addstr(gh + 1, 0, '+' + '-' * gw + '+')
        for y in range(1, gh + 1):
            self._safe_addstr(y, 0, '|')
            self._safe_addstr(y, gw + 1, '|')

        # Draw food
        for food in self.sim.food_items:
            fx, fy = food.position
            shape_name = food.get(FoodAttribute.SHAPE)
            color_name = food.get(FoodAttribute.COLOR)
            glyph = SHAPE_GLYPHS.get(shape_name, '?')
            pair = COLOR_PAIRS.get(color_name, 0)
            self._safe_addstr(fy + 1, fx + 1, glyph, curses.color_pair(pair))

        # Draw agents — count per cell, render top agent or count
        agent_cells = {}
        for agent in self.sim.population.agents:
            if agent.is_alive:
                pos = agent.position
                agent_cells[pos] = agent_cells.get(pos, 0) + 1

        for pos, count in agent_cells.items():
            x, y = pos
            if count == 1:
                self._safe_addstr(y + 1, x + 1, '@',
                                  curses.color_pair(AGENT_COLOR_PAIR) | curses.A_BOLD)
            else:
                label = str(min(count, 9))
                self._safe_addstr(y + 1, x + 1, label,
                                  curses.color_pair(AGENT_COLOR_PAIR) | curses.A_BOLD)

        # Stats panel
        info_y = gh + 2
        gen = self.sim.population.generation
        alive = self.sim.population.get_alive()
        n_alive = len(alive)
        n_total = self.sim.config.population_size

        status = 'PAUSED' if self.paused else 'RUNNING'
        self._safe_addstr(info_y, 0,
            f"{status} | Gen {gen} | Tick {self.tick_in_gen}/{self.sim.config.steps_per_generation}"
            f" | Alive {n_alive}/{n_total} | Food {len(self.sim.food_items)}")

        if alive:
            fitnesses = [a.fitness() for a in alive]
            mean_fit = sum(fitnesses) / len(fitnesses)
            max_fit = max(fitnesses)
            survival = n_alive / n_total
            self._safe_addstr(info_y + 1, 0,
                f"Fitness: mean={mean_fit:.1f} max={max_fit:.1f} | Survival={survival:.0%}")

            best = max(alive, key=lambda a: a.fitness())
            beliefs = best.get_belief_summary()
            if beliefs:
                items = list(beliefs.items())[:4]
                parts = [f"{k}: {v['mean']:+.1f}" for k, v in items]
                self._safe_addstr(info_y + 2, 0, f"Best beliefs: {' | '.join(parts)}")

        self._safe_addstr(info_y + 3, 0,
            "SPACE=pause  q=quit  +/-=speed  delay={:.0f}ms".format(self.delay * 1000))

        self.stdscr.refresh()

    def handle_input(self) -> bool:
        """Handle keyboard input. Returns False on quit."""
        key = self.stdscr.getch()
        if key == ord('q'):
            return False
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('+') or key == ord('='):
            self.delay = max(0.01, self.delay - 0.01)
        elif key == ord('-'):
            self.delay = min(0.5, self.delay + 0.01)
        return True

    def run(self) -> None:
        """Main loop: run generations with per-tick rendering."""
        cfg = self.sim.config
        for gen in range(cfg.num_generations):
            # Reset for generation
            self.sim.food_items = []
            for _ in range(cfg.max_food_items // 2):
                self.sim.spawn_food()

            # Randomise agent positions
            for agent in self.sim.population.agents:
                agent.position = (
                    random.randint(0, cfg.grid_size[0] - 1),
                    random.randint(0, cfg.grid_size[1] - 1),
                )

            for tick in range(cfg.steps_per_generation):
                if not self.handle_input():
                    return
                if not self.paused:
                    self.tick_in_gen = tick + 1
                    self.sim.step_tick()
                self.render()
                time.sleep(self.delay)

            # Selection and reproduction
            self.sim.population.run_selection(
                cfg.sexual_reproduction,
                initial_energy=cfg.initial_energy,
            )

            # Brief generation summary flash
            self.tick_in_gen = cfg.steps_per_generation
            self.render()
            time.sleep(0.3)

    def _safe_addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        """Write string to screen, ignoring out-of-bounds errors."""
        try:
            self.stdscr.addstr(y, x, text, attr)
        except curses.error:
            pass
