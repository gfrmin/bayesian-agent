#!/usr/bin/env python3
"""Entry point for Bayesian agent evolutionary simulation.

Interactive mode (default): curses visualization of population foraging.
Batch mode (--batch): text-only evolutionary run with summary output.
"""

import argparse
import curses
import sys

from simulation import Simulation, SimulationConfig, run_experiment
from visualization import PopulationVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Evolutionary Bayesian Agents")
    p.add_argument("--batch", action="store_true",
                   help="Run batch evolution (text output, no curses)")
    p.add_argument("--reality", default="simple",
                   choices=["simple", "interaction", "hidden", "temporal", "hierarchical"],
                   help="Reality preset (default: simple)")
    p.add_argument("--population", type=int, default=30,
                   help="Population size (default: 30)")
    p.add_argument("--generations", type=int, default=50,
                   help="Number of generations (default: 50)")
    return p.parse_args()


def print_final_stats(sim: Simulation) -> None:
    """Print summary after evolution completes."""
    if not hasattr(sim, "best_ever") or not sim.best_ever:
        print("\nNo surviving agents — evolution did not find a solution.")
        return

    best = sim.best_ever
    print(f"\n{'=' * 60}")
    print("EVOLUTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best agent found at generation {best['generation_found']}")
    print(f"Fitness: {best['fitness']:.2f}")
    print(f"Foods eaten: {best['foods_eaten']}")
    print(f"\nSensors evolved: {best['sensors']}")
    print(f"BN structure: {best['bn_structure']}")
    print(f"\nLearned beliefs:")
    if best["beliefs"]:
        for config, belief in best["beliefs"].items():
            print(f"  {config}: \u03bc={belief['mean']:.2f}, \u03c3={belief['std']:.2f} "
                  f"(n={belief['observations']})")
    else:
        print("  (No configurations observed)")


def run_interactive(args) -> None:
    """Launch curses visualization."""
    config = SimulationConfig(
        reality_type=args.reality,
        population_size=args.population,
        num_generations=args.generations,
    )
    sim = Simulation(config)

    curses.wrapper(lambda stdscr: PopulationVisualizer(stdscr, sim).run())
    print_final_stats(sim)


def run_batch(args) -> None:
    """Run headless evolutionary batch."""
    run_experiment(
        reality_type=args.reality,
        num_generations=args.generations,
        population_size=args.population,
        verbose=True,
    )


if __name__ == "__main__":
    try:
        args = parse_args()
        if args.batch:
            run_batch(args)
        else:
            run_interactive(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
