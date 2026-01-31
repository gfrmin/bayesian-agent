"""
Bayesian Agent: An agent that learns within its lifetime.

The agent's genome specifies:
- Which variables it can perceive (sensors)
- Which variables it believes are relevant (BN structure)
- Prior beliefs (hyperparameters)

Learning fills in:
- Conditional probability distributions P(Energy | observed attributes)

The key insight: the genome specifies the FORM of knowledge,
learning fills in the CONTENT.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, FrozenSet
import random
import math
from collections import defaultdict

from reality import FoodItem, FoodAttribute, Reality
from cognitive_genome import CognitiveGenome, BayesianNetworkStructure


@dataclass
class NormalBelief:
    """
    A belief represented as a Normal distribution.
    
    Tracks:
    - mean: Expected value
    - variance: Uncertainty
    - n: Effective sample size (pseudo-observations)
    - observations: Actual number of data points seen
    """
    mean: float
    variance: float
    n: float  # Pseudo-observation count
    observations: int = 0
    
    @property
    def std(self) -> float:
        return math.sqrt(max(0.001, self.variance))
    
    def update(self, observation: float) -> None:
        """Bayesian update assuming known variance (Normal-Normal conjugacy)."""
        # Posterior mean: weighted average of prior and observation
        self.mean = (self.n * self.mean + observation) / (self.n + 1)
        
        # Posterior variance: shrinks with more data
        self.variance = self.variance / (1 + 1/self.n)
        
        # Update counts
        self.n += 1
        self.observations += 1
    
    def sample(self) -> float:
        """Thompson sampling: draw from posterior."""
        return random.gauss(self.mean, self.std)
    
    def expected_value(self) -> float:
        """Point estimate (posterior mean)."""
        return self.mean
    
    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """Approximate confidence interval."""
        return (self.mean - z * self.std, self.mean + z * self.std)


def config_to_key(config: Dict[FoodAttribute, Any]) -> Tuple:
    """Convert a configuration dict to a hashable key."""
    return tuple(sorted((k.name, v) for k, v in config.items()))


@dataclass
class BayesianAgent:
    """
    An agent that learns about its world through Bayesian inference.
    
    The agent maintains beliefs about P(Energy | configuration) where
    'configuration' is an assignment of values to the variables in its
    BN structure.
    
    Example:
    -------
    If the agent's BN has edges SHAPE→ENERGY and COLOR→ENERGY,
    then it maintains separate beliefs for each (shape, color) pair:
    - P(Energy | shape=circle, color=red)
    - P(Energy | shape=circle, color=green)
    - ... etc.
    
    This is a tabular representation. More sophisticated agents could use
    parametric models or function approximation.
    """
    
    genome: CognitiveGenome
    position: Tuple[int, int] = (0, 0)
    energy: float = 10.0
    
    # Learned beliefs: config_key -> NormalBelief
    beliefs: Dict[Tuple, NormalBelief] = field(default_factory=dict)
    
    # Fitness tracking
    total_energy_gained: float = 0.0
    total_energy_spent: float = 0.0
    foods_eaten: int = 0
    steps_taken: int = 0
    
    # State
    is_alive: bool = True
    
    def __post_init__(self):
        self.agent_id = random.getrandbits(64)
        self._relevant_attributes = self.genome.bn_structure.get_parents_of_energy()
    
    def perceive(self, food: FoodItem) -> Dict[FoodAttribute, Any]:
        """
        Extract the attributes of food that this agent can see.
        
        Filtered by the agent's sensor suite.
        """
        return food.get_perceptible(self.genome.sensors)
    
    def get_relevant_config(self, food: FoodItem) -> Dict[FoodAttribute, Any]:
        """
        Extract only the attributes that are in the agent's BN.
        
        This is what the agent uses for prediction — the variables
        it believes are relevant to energy.
        """
        perceived = self.perceive(food)
        return {
            attr: val for attr, val in perceived.items()
            if attr in self._relevant_attributes
        }
    
    def get_or_create_belief(self, config: Dict[FoodAttribute, Any]) -> NormalBelief:
        """
        Get existing belief or initialise from prior.
        
        The prior comes from the genome.
        """
        key = config_to_key(config)
        
        if key not in self.beliefs:
            prior = self.genome.prior
            
            # Uncertainty bonus: more uncertain about novel configurations
            initial_mean = prior.global_mean
            if prior.uncertainty_bonus != 0:
                # Optimistic/pessimistic about novel things
                initial_mean += prior.uncertainty_bonus
            
            self.beliefs[key] = NormalBelief(
                mean=initial_mean,
                variance=prior.global_variance,
                n=prior.pseudo_observations,
            )
        
        return self.beliefs[key]
    
    def predict_energy(self, food: FoodItem) -> NormalBelief:
        """
        Predict energy value for a food item.
        
        Returns the belief distribution, not just a point estimate.
        """
        config = self.get_relevant_config(food)
        
        if not config:
            # Agent models nothing as relevant — use global prior
            prior = self.genome.prior
            return NormalBelief(
                mean=prior.global_mean + prior.uncertainty_bonus,
                variance=prior.global_variance,
                n=prior.pseudo_observations,
            )
        
        return self.get_or_create_belief(config)
    
    def evaluate_food(self, food: FoodItem, distance: float, 
                      movement_cost: float = 0.1) -> float:
        """
        Compute subjective value of pursuing a food item.
        
        Uses Thompson sampling for exploration-exploitation balance.
        """
        belief = self.predict_energy(food)
        
        # Thompson sampling: draw from posterior
        sampled_energy = belief.sample()
        
        # Subtract expected cost to reach food
        expected_cost = distance * movement_cost
        
        return sampled_energy - expected_cost
    
    def learn(self, food: FoodItem, actual_energy: float) -> None:
        """
        Update beliefs after observing the outcome of eating food.
        
        This is the within-lifetime learning.
        """
        config = self.get_relevant_config(food)
        
        if config:
            belief = self.get_or_create_belief(config)
            belief.update(actual_energy)
        
        # Track statistics
        self.foods_eaten += 1
        if actual_energy > 0:
            self.total_energy_gained += actual_energy
    
    def select_food(self, available_foods: List[Tuple[FoodItem, float]],
                    movement_cost: float = 0.1) -> Optional[FoodItem]:
        """
        Select which food to pursue from available options.
        
        Args:
            available_foods: List of (food, distance) tuples
            movement_cost: Energy cost per unit distance
            
        Returns:
            The selected food item, or None if none are worth pursuing
        """
        if not available_foods:
            return None
        
        best_food = None
        best_value = float('-inf')
        
        for food, distance in available_foods:
            value = self.evaluate_food(food, distance, movement_cost)
            if value > best_value:
                best_value = value
                best_food = food
        
        # Only pursue if expected value is positive
        if best_value > 0:
            return best_food
        
        return None
    
    def move_towards(self, target: Tuple[int, int], 
                     movement_cost: float = 0.1) -> bool:
        """
        Move one step towards target position.
        
        Returns True if moved, False if already at target.
        """
        if self.position == target:
            return False
        
        dx = 0 if target[0] == self.position[0] else (1 if target[0] > self.position[0] else -1)
        dy = 0 if target[1] == self.position[1] else (1 if target[1] > self.position[1] else -1)
        
        self.position = (self.position[0] + dx, self.position[1] + dy)
        self.energy -= movement_cost
        self.total_energy_spent += movement_cost
        self.steps_taken += 1
        
        if self.energy <= 0:
            self.is_alive = False
        
        return True
    
    def consume(self, food: FoodItem, actual_energy: float) -> None:
        """
        Consume food and learn from the experience.
        """
        self.energy += actual_energy
        self.learn(food, actual_energy)
        
        if self.energy <= 0:
            self.is_alive = False
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Summarise current beliefs for display/logging."""
        summary = {}
        for key, belief in self.beliefs.items():
            config_str = ', '.join(f"{k}={v}" for k, v in key)
            summary[config_str] = {
                'mean': round(belief.mean, 2),
                'std': round(belief.std, 2),
                'observations': belief.observations,
            }
        return summary
    
    def fitness(self) -> float:
        """
        Compute fitness score for selection.
        
        Multiple possible fitness functions:
        - Total energy gained
        - Survival time
        - Net energy (gained - spent)
        - Efficiency (gained / spent)
        """
        # Default: net energy gained (rewards efficient foraging)
        return self.total_energy_gained - self.total_energy_spent
    
    def create_offspring(self, mate: Optional['BayesianAgent'] = None) -> 'BayesianAgent':
        """
        Create a child agent.
        
        If mate is provided, uses sexual reproduction (crossover + mutation).
        Otherwise, uses asexual reproduction (mutation only).
        """
        if mate is not None:
            child_genome = CognitiveGenome.crossover(self.genome, mate.genome)
        else:
            child_genome = self.genome.mutate()
        
        return BayesianAgent(
            genome=child_genome,
            position=self.position,  # Child starts where parent is
            energy=10.0,  # Fresh energy
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialise agent state for logging."""
        return {
            'agent_id': self.agent_id,
            'genome': self.genome.to_dict(),
            'position': self.position,
            'energy': self.energy,
            'fitness': self.fitness(),
            'foods_eaten': self.foods_eaten,
            'steps_taken': self.steps_taken,
            'is_alive': self.is_alive,
            'num_beliefs': len(self.beliefs),
            'belief_summary': self.get_belief_summary(),
        }
    
    def __repr__(self):
        status = "alive" if self.is_alive else "dead"
        return (f"Agent[{status}, energy={self.energy:.1f}, "
                f"fitness={self.fitness():.1f}, {self.genome}]")


class Population:
    """
    A population of agents undergoing evolution.
    
    Manages:
    - Selection (who reproduces)
    - Reproduction (creating offspring)
    - Generational tracking
    """
    
    def __init__(self, 
                 size: int = 50,
                 initial_genome_factory=None,
                 selection_pressure: float = 0.5,
                 initial_energy: float = 15.0):
        """
        Args:
            size: Target population size
            initial_genome_factory: Callable that returns a CognitiveGenome
            selection_pressure: Fraction of population that reproduces (top N%)
            initial_energy: Starting energy for agents
        """
        self.target_size = size
        self.selection_pressure = selection_pressure
        self.initial_energy = initial_energy
        self.generation = 0
        
        # Initialise population
        if initial_genome_factory is None:
            from cognitive_genome import standard_genome
            initial_genome_factory = standard_genome
        
        self.agents: List[BayesianAgent] = []
        for _ in range(size):
            genome = initial_genome_factory()
            self.agents.append(BayesianAgent(genome=genome, energy=initial_energy))
        
        # Statistics
        self.history: List[Dict[str, Any]] = []
    
    def get_alive(self) -> List[BayesianAgent]:
        """Return all living agents."""
        return [a for a in self.agents if a.is_alive]
    
    def run_selection(self, sexual: bool = True, 
                      initial_energy: float = 15.0) -> List[BayesianAgent]:
        """
        Select agents for reproduction and create next generation.
        
        Args:
            sexual: If True, pairs agents for crossover. If False, asexual.
            initial_energy: Starting energy for new agents.
        """
        alive = self.get_alive()
        
        if not alive:
            # Extinction! Start fresh.
            from cognitive_genome import standard_genome
            return [BayesianAgent(genome=standard_genome(), energy=initial_energy) 
                    for _ in range(self.target_size)]
        
        # Sort by fitness
        alive.sort(key=lambda a: a.fitness(), reverse=True)
        
        # Select top fraction
        n_parents = max(2, int(len(alive) * self.selection_pressure))
        parents = alive[:n_parents]
        
        # Record statistics before reproduction
        self._record_generation_stats(alive, parents)
        
        # Create next generation
        offspring = []
        while len(offspring) < self.target_size:
            if sexual and len(parents) >= 2:
                # Sexual reproduction
                p1, p2 = random.sample(parents, 2)
                child = p1.create_offspring(p2)
            else:
                # Asexual reproduction
                parent = random.choice(parents)
                child = parent.create_offspring()
            
            child.energy = initial_energy  # Reset energy for new generation
            offspring.append(child)
        
        self.generation += 1
        self.agents = offspring
        return offspring
    
    def _record_generation_stats(self, alive: List[BayesianAgent], 
                                  parents: List[BayesianAgent]) -> None:
        """Record statistics for this generation."""
        if not alive:
            return
        
        fitnesses = [a.fitness() for a in alive]
        
        # Genome diversity: count unique BN structures
        structures = set()
        sensor_sets = set()
        for a in alive:
            structures.add(frozenset(str(e) for e in a.genome.bn_structure.edges))
            sensor_sets.add(a.genome.sensors)
        
        stats = {
            'generation': self.generation,
            'population_size': len(alive),
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'fitness_std': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5,
            'num_unique_structures': len(structures),
            'num_unique_sensor_sets': len(sensor_sets),
            'mean_complexity': sum(a.genome.complexity_cost() for a in alive) / len(alive),
            'survival_rate': len(alive) / self.target_size,
        }
        
        self.history.append(stats)
    
    def get_best_agent(self) -> Optional[BayesianAgent]:
        """Return the highest-fitness living agent."""
        alive = self.get_alive()
        if not alive:
            return None
        return max(alive, key=lambda a: a.fitness())
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """Analyse genetic diversity in the population."""
        alive = self.get_alive()
        if not alive:
            return {'diversity': 0}
        
        # Count unique genomes
        genomes = {}
        for a in alive:
            key = (a.genome.sensors, 
                   frozenset(str(e) for e in a.genome.bn_structure.edges))
            genomes[key] = genomes.get(key, 0) + 1
        
        return {
            'num_unique_genomes': len(genomes),
            'most_common_genome_freq': max(genomes.values()) / len(alive),
            'genome_frequencies': genomes,
        }
