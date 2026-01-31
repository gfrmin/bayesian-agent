"""
Simulation: The driver that runs the evolutionary process.

This module orchestrates:
1. Environment setup (Reality)
2. Agent lifecycle (birth, learning, death)
3. Selection and reproduction
4. Data collection for analysis

The "god" (user) controls:
- Which Reality to use (the problem)
- Selection criteria (what counts as fitness)
- Simulation parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import random
import math
import json
from datetime import datetime

from reality import Reality, FoodItem, FoodAttribute, REALITY_PRESETS
from cognitive_genome import CognitiveGenome, GENOME_PRESETS
from bayesian_agent import BayesianAgent, Population


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    
    # Environment
    reality_type: str = 'simple'
    grid_size: Tuple[int, int] = (15, 15)  # Smaller grid = denser food
    food_spawn_rate: float = 0.5  # Higher spawn rate
    max_food_items: int = 40  # More food available
    
    # Population
    population_size: int = 50
    initial_genome_type: str = 'standard'
    selection_pressure: float = 0.3  # Top 30% reproduce
    sexual_reproduction: bool = True
    
    # Agent parameters
    initial_energy: float = 15.0  # More starting energy
    movement_cost: float = 0.05  # Lower movement cost
    observation_radius: int = 6  # Can see further
    
    # Simulation
    steps_per_generation: int = 150  # Shorter generations
    num_generations: int = 100
    
    # Logging
    log_interval: int = 10
    save_history: bool = True


class Simulation:
    """
    The main simulation driver.
    
    Runs evolutionary dynamics:
    1. Agents forage in the environment
    2. Successful agents reproduce
    3. Offspring inherit (mutated) cognitive architectures
    4. Repeat
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # Initialise reality (the problem to solve)
        reality_factory = REALITY_PRESETS.get(
            self.config.reality_type, 
            REALITY_PRESETS['simple']
        )
        self.reality = reality_factory()
        self.reality.grid_size = self.config.grid_size
        
        # Initialise population
        genome_factory = GENOME_PRESETS.get(
            self.config.initial_genome_type,
            GENOME_PRESETS['standard']
        )
        self.population = Population(
            size=self.config.population_size,
            initial_genome_factory=genome_factory,
            selection_pressure=self.config.selection_pressure,
            initial_energy=self.config.initial_energy,
        )
        
        # Food in the environment
        self.food_items: List[FoodItem] = []
        
        # Statistics
        self.step_count = 0
        self.generation_stats: List[Dict[str, Any]] = []
        
    def spawn_food(self) -> None:
        """Probabilistically spawn new food in the environment."""
        if len(self.food_items) >= self.config.max_food_items:
            return
        
        if random.random() < self.config.food_spawn_rate:
            food = self.reality.spawn_food()
            self.food_items.append(food)
    
    def get_nearby_food(self, position: Tuple[int, int], 
                        radius: int) -> List[Tuple[FoodItem, float]]:
        """Get food items within radius of position, with distances."""
        nearby = []
        for food in self.food_items:
            dx = food.position[0] - position[0]
            dy = food.position[1] - position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance <= radius:
                nearby.append((food, distance))
        return nearby
    
    def remove_food(self, food: FoodItem) -> None:
        """Remove a food item from the environment."""
        if food in self.food_items:
            self.food_items.remove(food)
    
    def step_agent(self, agent: BayesianAgent) -> None:
        """Execute one step for a single agent."""
        if not agent.is_alive:
            return
        
        # Get nearby food
        nearby = self.get_nearby_food(
            agent.position, 
            self.config.observation_radius
        )
        
        # Select target
        target_food = agent.select_food(nearby, self.config.movement_cost)
        
        if target_food is not None:
            # Move towards food
            if agent.position != target_food.position:
                agent.move_towards(target_food.position, self.config.movement_cost)
            
            # If at food, consume it
            if agent.position == target_food.position:
                actual_energy = self.reality.compute_true_energy(target_food)
                agent.consume(target_food, actual_energy)
                self.remove_food(target_food)
        else:
            # Random walk if no good food
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            new_x = max(0, min(self.config.grid_size[0]-1, agent.position[0] + dx))
            new_y = max(0, min(self.config.grid_size[1]-1, agent.position[1] + dy))
            if (new_x, new_y) != agent.position:
                agent.position = (new_x, new_y)
                agent.energy -= self.config.movement_cost
                agent.total_energy_spent += self.config.movement_cost
                agent.steps_taken += 1
                if agent.energy <= 0:
                    agent.is_alive = False
    
    def step_tick(self) -> None:
        """Advance one simulation tick (food spawn + all agents act)."""
        self.step_count += 1
        self.reality.step()
        self.spawn_food()
        for agent in self.population.agents:
            self.step_agent(agent)

    def run_generation(self) -> Dict[str, Any]:
        """Run one generation of the simulation."""
        # Reset food
        self.food_items = []
        for _ in range(self.config.max_food_items // 2):
            self.spawn_food()

        # Run simulation steps
        for step in range(self.config.steps_per_generation):
            self.step_tick()
        
        # Collect generation statistics
        alive = self.population.get_alive()
        stats = {
            'generation': self.population.generation,
            'alive_count': len(alive),
            'survival_rate': len(alive) / self.config.population_size,
        }
        
        if alive:
            fitnesses = [a.fitness() for a in alive]
            stats.update({
                'mean_fitness': sum(fitnesses) / len(fitnesses),
                'max_fitness': max(fitnesses),
                'min_fitness': min(fitnesses),
                'best_agent': self.population.get_best_agent().to_dict(),
            })
            
            # Genome statistics
            sensor_counts = {}
            edge_counts = {}
            for a in alive:
                for s in a.genome.sensors:
                    sensor_counts[s.name] = sensor_counts.get(s.name, 0) + 1
                for e in a.genome.bn_structure.edges:
                    edge_counts[str(e)] = edge_counts.get(str(e), 0) + 1
            
            stats['sensor_frequencies'] = {
                k: v/len(alive) for k, v in sensor_counts.items()
            }
            stats['edge_frequencies'] = {
                k: v/len(alive) for k, v in edge_counts.items()
            }
        
        return stats
    
    def run_evolution(self, 
                      callback: Callable[[int, Dict], None] = None) -> List[Dict]:
        """
        Run the full evolutionary process.
        
        Args:
            callback: Optional function called each generation with (gen, stats)
        """
        all_stats = []
        best_ever = None
        best_ever_fitness = float('-inf')
        
        for gen in range(self.config.num_generations):
            # Run this generation
            stats = self.run_generation()
            all_stats.append(stats)
            
            # Track best agent BEFORE reproduction (so we capture learned beliefs)
            current_best = self.population.get_best_agent()
            if current_best and current_best.fitness() > best_ever_fitness:
                best_ever_fitness = current_best.fitness()
                # Deep copy the relevant info since agent will be replaced
                best_ever = {
                    'fitness': current_best.fitness(),
                    'genome': current_best.genome.to_dict(),
                    'genome_repr': repr(current_best.genome),
                    'sensors': [s.name for s in current_best.genome.sensors],
                    'bn_structure': str(current_best.genome.bn_structure),
                    'beliefs': current_best.get_belief_summary(),
                    'foods_eaten': current_best.foods_eaten,
                    'generation_found': gen,
                }
            
            # Callback for logging/display
            if callback:
                callback(gen, stats)
            
            # Selection and reproduction
            self.population.run_selection(
                self.config.sexual_reproduction,
                initial_energy=self.config.initial_energy
            )
            
            # Reset agent positions for next generation
            for agent in self.population.agents:
                agent.position = (
                    random.randint(0, self.config.grid_size[0]-1),
                    random.randint(0, self.config.grid_size[1]-1)
                )
        
        self.generation_stats = all_stats
        self.best_ever = best_ever
        return all_stats
    
    def get_evolved_genome(self) -> Optional[CognitiveGenome]:
        """Get the genome of the best agent after evolution."""
        best = self.population.get_best_agent()
        return best.genome if best else None
    
    def save_results(self, filename: str) -> None:
        """Save simulation results to JSON."""
        results = {
            'config': {
                'reality_type': self.config.reality_type,
                'population_size': self.config.population_size,
                'num_generations': self.config.num_generations,
                'steps_per_generation': self.config.steps_per_generation,
            },
            'generation_stats': self.generation_stats,
            'final_population': [a.to_dict() for a in self.population.get_alive()],
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)


def run_experiment(reality_type: str = 'simple',
                   num_generations: int = 50,
                   population_size: int = 30,
                   verbose: bool = True) -> Simulation:
    """
    Convenience function to run an experiment.
    
    Returns the Simulation object for further analysis.
    """
    config = SimulationConfig(
        reality_type=reality_type,
        num_generations=num_generations,
        population_size=population_size,
    )
    
    sim = Simulation(config)
    
    def log_progress(gen: int, stats: Dict):
        if verbose and gen % 5 == 0:
            print(f"Gen {gen:3d}: {stats.get('alive_count', 0):2d} alive, "
                  f"fitness={stats.get('mean_fitness', 0):.2f} "
                  f"(max={stats.get('max_fitness', 0):.2f})")
    
    print(f"\nRunning evolution in '{reality_type}' reality...")
    print(f"Population: {population_size}, Generations: {num_generations}\n")
    
    sim.run_evolution(callback=log_progress)
    
    # Report final state
    if hasattr(sim, 'best_ever') and sim.best_ever:
        best = sim.best_ever
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Best agent found at generation {best['generation_found']}")
        print(f"Fitness: {best['fitness']:.2f}")
        print(f"Foods eaten: {best['foods_eaten']}")
        print(f"\nSensors evolved: {best['sensors']}")
        print(f"BN structure: {best['bn_structure']}")
        print(f"\nLearned beliefs (what the agent figured out):")
        if best['beliefs']:
            for config, belief in best['beliefs'].items():
                print(f"  {config}: μ={belief['mean']:.2f}, σ={belief['std']:.2f} "
                      f"(n={belief['observations']})")
        else:
            print("  (No configurations observed)")
    
    return sim


# =============================================================================
# MAIN: Run a demonstration
# =============================================================================

if __name__ == '__main__':
    # Run experiments on different realities
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: Simple Reality (independent shape + color effects)")
    print("="*70)
    sim1 = run_experiment('simple', num_generations=30, verbose=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Interaction Reality (shape-color interactions)")
    print("="*70)
    sim2 = run_experiment('interaction', num_generations=30, verbose=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: Hierarchical Reality (complex dependencies)")
    print("="*70)
    sim3 = run_experiment('hierarchical', num_generations=50, verbose=True)
