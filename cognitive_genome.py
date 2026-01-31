"""
Cognitive Genome: The heritable specification of an agent's cognitive architecture.

This encodes:
1. SENSORS: Which attributes of reality the agent can perceive
2. STRUCTURE: The Bayesian network topology (which variables depend on which)
3. PRIORS: Initial beliefs before any learning

Crucially, the genome does NOT encode:
- The actual probability values (those are learned)
- What to believe (Lamarckian inheritance is forbidden)

The BN structure in the genome represents the agent's "theory of the world" —
its assumptions about what causes what. Evolution discovers which theories
are predictively useful in a given reality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, FrozenSet, Any
from enum import Enum, auto
import random
import copy
import math
from reality import FoodAttribute, PERCEPTIBLE_ATTRIBUTES


@dataclass(frozen=True)
class BNEdge:
    """
    A directed edge in a Bayesian network.
    
    parent -> child means: P(child | parent, ...) depends on parent
    """
    parent: FoodAttribute
    child: str  # Either another FoodAttribute name or 'ENERGY'
    
    def __repr__(self):
        return f"{self.parent.name} → {self.child}"


@dataclass
class BayesianNetworkStructure:
    """
    The structure (topology) of an agent's internal world model.
    
    The BN always has 'ENERGY' as the final node (what we're predicting).
    The structure specifies which perceived attributes connect to ENERGY
    and whether there are any intermediate dependencies.
    
    For simplicity, we start with a two-layer model:
    - Layer 1: Perceived attributes (observations)
    - Layer 2: ENERGY (prediction target)
    
    Edges go from Layer 1 to Layer 2, specifying which attributes
    the agent believes are relevant to predicting energy.
    
    Future extensions could add:
    - Intermediate latent variables
    - Edges between observables (to model correlations)
    """
    
    edges: FrozenSet[BNEdge] = field(default_factory=frozenset)
    
    def get_parents_of_energy(self) -> Set[FoodAttribute]:
        """Which attributes does this agent think predict energy?"""
        return {e.parent for e in self.edges if e.child == 'ENERGY'}
    
    def add_edge(self, parent: FoodAttribute) -> 'BayesianNetworkStructure':
        """Add an edge from an attribute to ENERGY."""
        new_edges = set(self.edges)
        new_edges.add(BNEdge(parent, 'ENERGY'))
        return BayesianNetworkStructure(frozenset(new_edges))
    
    def remove_edge(self, parent: FoodAttribute) -> 'BayesianNetworkStructure':
        """Remove an edge from an attribute to ENERGY."""
        new_edges = {e for e in self.edges if e.parent != parent}
        return BayesianNetworkStructure(frozenset(new_edges))
    
    def complexity(self) -> int:
        """Number of edges (proxy for model complexity)."""
        return len(self.edges)
    
    def __repr__(self):
        parents = self.get_parents_of_energy()
        if not parents:
            return "BN: [nothing] → ENERGY"
        return f"BN: {{{', '.join(p.name for p in parents)}}} → ENERGY"


@dataclass
class CognitivePrior:
    """
    Prior beliefs about conditional distributions in the BN.
    
    For each configuration of parent values, we have a prior
    over the energy distribution (assumed Normal for simplicity).
    
    We use a hierarchical structure:
    - Global prior mean and variance
    - Per-configuration adjustments (initially zero)
    """
    global_mean: float = 0.0
    global_variance: float = 10.0
    pseudo_observations: float = 0.1  # Prior strength
    
    # Optimism/pessimism: shift prior mean based on uncertainty
    uncertainty_bonus: float = 0.0  # Positive = optimistic about unknowns


@dataclass
class CognitiveGenome:
    """
    Complete genetic specification of an agent's cognitive architecture.
    
    Components:
    ----------
    sensors: Set of attributes this agent can perceive
    bn_structure: Which perceived attributes the agent models as relevant
    prior: Prior belief parameters
    
    Constraints:
    -----------
    - Can only have BN edges from attributes in sensor set
    - BN structure determines the "theory"; learning fills in parameters
    
    Mutation Operators:
    ------------------
    - Add/remove sensor
    - Add/remove BN edge
    - Adjust prior parameters
    """
    
    sensors: FrozenSet[FoodAttribute] = field(
        default_factory=lambda: frozenset({FoodAttribute.SHAPE, FoodAttribute.COLOR})
    )
    bn_structure: BayesianNetworkStructure = field(
        default_factory=BayesianNetworkStructure
    )
    prior: CognitivePrior = field(default_factory=CognitivePrior)
    
    # Mutation rates
    sensor_mutation_rate: float = 0.1
    edge_mutation_rate: float = 0.15
    prior_mutation_rate: float = 0.1
    prior_mutation_scale: float = 0.2
    
    def __post_init__(self):
        self.genome_id = random.getrandbits(64)
        self.parent_ids: Tuple[Optional[int], Optional[int]] = (None, None)
        self.generation: int = 0
        
        # Ensure BN structure only references available sensors
        self._validate_structure()
    
    def _validate_structure(self):
        """Ensure BN edges only reference sensors this agent has."""
        valid_edges = frozenset(
            e for e in self.bn_structure.edges 
            if e.parent in self.sensors
        )
        if valid_edges != self.bn_structure.edges:
            self.bn_structure = BayesianNetworkStructure(valid_edges)
    
    def can_perceive(self, attr: FoodAttribute) -> bool:
        """Check if this agent can see a given attribute."""
        return attr in self.sensors and attr in PERCEPTIBLE_ATTRIBUTES
    
    def models_as_relevant(self, attr: FoodAttribute) -> bool:
        """Check if this agent's theory includes this attribute."""
        return attr in self.bn_structure.get_parents_of_energy()
    
    def mutate(self) -> 'CognitiveGenome':
        """Create a mutated copy of this genome."""
        new_sensors = set(self.sensors)
        new_structure = self.bn_structure
        new_prior = copy.deepcopy(self.prior)
        
        # Sensor mutations
        if random.random() < self.sensor_mutation_rate:
            # Add or remove a sensor
            available = PERCEPTIBLE_ATTRIBUTES - new_sensors
            current = new_sensors & PERCEPTIBLE_ATTRIBUTES
            
            if available and (not current or random.random() < 0.5):
                # Add a sensor
                new_sensor = random.choice(list(available))
                new_sensors.add(new_sensor)
            elif current:
                # Remove a sensor
                removed = random.choice(list(current))
                new_sensors.discard(removed)
                # Also remove any BN edges involving this sensor
                new_structure = BayesianNetworkStructure(
                    frozenset(e for e in new_structure.edges if e.parent != removed)
                )
        
        # BN structure mutations
        if random.random() < self.edge_mutation_rate:
            current_parents = new_structure.get_parents_of_energy()
            unused_sensors = new_sensors - current_parents
            
            if unused_sensors and (not current_parents or random.random() < 0.5):
                # Add an edge
                new_parent = random.choice(list(unused_sensors))
                new_structure = new_structure.add_edge(new_parent)
            elif current_parents:
                # Remove an edge
                removed = random.choice(list(current_parents))
                new_structure = new_structure.remove_edge(removed)
        
        # Prior parameter mutations
        if random.random() < self.prior_mutation_rate:
            new_prior.global_mean += random.gauss(0, self.prior_mutation_scale * 5)
            new_prior.global_mean = max(-10, min(10, new_prior.global_mean))
        
        if random.random() < self.prior_mutation_rate:
            new_prior.global_variance *= math.exp(random.gauss(0, self.prior_mutation_scale))
            new_prior.global_variance = max(0.1, min(100, new_prior.global_variance))
        
        if random.random() < self.prior_mutation_rate:
            new_prior.pseudo_observations *= math.exp(random.gauss(0, self.prior_mutation_scale))
            new_prior.pseudo_observations = max(0.01, min(10, new_prior.pseudo_observations))
        
        if random.random() < self.prior_mutation_rate:
            new_prior.uncertainty_bonus += random.gauss(0, self.prior_mutation_scale * 2)
            new_prior.uncertainty_bonus = max(-5, min(5, new_prior.uncertainty_bonus))
        
        child = CognitiveGenome(
            sensors=frozenset(new_sensors),
            bn_structure=new_structure,
            prior=new_prior,
            sensor_mutation_rate=self.sensor_mutation_rate,
            edge_mutation_rate=self.edge_mutation_rate,
            prior_mutation_rate=self.prior_mutation_rate,
            prior_mutation_scale=self.prior_mutation_scale,
        )
        child.parent_ids = (self.genome_id, None)
        child.generation = self.generation + 1
        return child
    
    @classmethod
    def crossover(cls, parent1: 'CognitiveGenome', parent2: 'CognitiveGenome') -> 'CognitiveGenome':
        """Sexual reproduction: combine genetic material from two parents."""
        # Sensors: union with random dropout
        all_sensors = parent1.sensors | parent2.sensors
        shared_sensors = parent1.sensors & parent2.sensors
        unique_sensors = all_sensors - shared_sensors
        
        new_sensors = set(shared_sensors)  # Keep shared sensors
        for sensor in unique_sensors:
            if random.random() < 0.5:  # 50% chance to inherit unique sensors
                new_sensors.add(sensor)
        
        # BN structure: combine edges, keeping only those for inherited sensors
        all_edges = parent1.bn_structure.edges | parent2.bn_structure.edges
        valid_edges = frozenset(e for e in all_edges if e.parent in new_sensors)
        new_structure = BayesianNetworkStructure(valid_edges)
        
        # Prior: interpolate between parents
        weight = random.random()
        new_prior = CognitivePrior(
            global_mean=weight * parent1.prior.global_mean + (1-weight) * parent2.prior.global_mean,
            global_variance=math.exp(weight * math.log(parent1.prior.global_variance) + 
                                     (1-weight) * math.log(parent2.prior.global_variance)),
            pseudo_observations=math.exp(weight * math.log(parent1.prior.pseudo_observations) + 
                                         (1-weight) * math.log(parent2.prior.pseudo_observations)),
            uncertainty_bonus=weight * parent1.prior.uncertainty_bonus + (1-weight) * parent2.prior.uncertainty_bonus,
        )
        
        child = cls(
            sensors=frozenset(new_sensors),
            bn_structure=new_structure,
            prior=new_prior,
        )
        child.parent_ids = (parent1.genome_id, parent2.genome_id)
        child.generation = max(parent1.generation, parent2.generation) + 1
        
        # Apply mutation to offspring
        return child.mutate()
    
    def complexity_cost(self) -> float:
        """
        Compute a complexity measure (could be used for regularisation).
        
        More sensors + more edges = more complex theory.
        Not used as a direct fitness penalty (time is the only cost),
        but useful for analysis.
        """
        return len(self.sensors) + self.bn_structure.complexity()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialise for logging."""
        return {
            'genome_id': self.genome_id,
            'parent_ids': self.parent_ids,
            'generation': self.generation,
            'sensors': [s.name for s in self.sensors],
            'bn_edges': [str(e) for e in self.bn_structure.edges],
            'prior_mean': self.prior.global_mean,
            'prior_variance': self.prior.global_variance,
            'complexity': self.complexity_cost(),
        }
    
    def __repr__(self):
        sensors_str = ', '.join(s.name for s in self.sensors)
        return f"Genome[sensors={{{sensors_str}}}, {self.bn_structure}]"


# =============================================================================
# PRESET GENOMES: Starting configurations
# =============================================================================

def minimal_genome() -> CognitiveGenome:
    """Minimal agent: perceives only shape, models only shape."""
    g = CognitiveGenome(
        sensors=frozenset({FoodAttribute.SHAPE}),
        bn_structure=BayesianNetworkStructure(
            frozenset({BNEdge(FoodAttribute.SHAPE, 'ENERGY')})
        ),
    )
    return g


def standard_genome() -> CognitiveGenome:
    """Standard agent: perceives shape+color, models both."""
    g = CognitiveGenome(
        sensors=frozenset({FoodAttribute.SHAPE, FoodAttribute.COLOR}),
        bn_structure=BayesianNetworkStructure(
            frozenset({
                BNEdge(FoodAttribute.SHAPE, 'ENERGY'),
                BNEdge(FoodAttribute.COLOR, 'ENERGY'),
            })
        ),
    )
    return g


def rich_genome() -> CognitiveGenome:
    """Rich agent: many sensors, models all of them."""
    sensors = frozenset({
        FoodAttribute.SHAPE, FoodAttribute.COLOR, FoodAttribute.SIZE,
        FoodAttribute.TEXTURE, FoodAttribute.TEMPERATURE,
    })
    edges = frozenset(BNEdge(s, 'ENERGY') for s in sensors)
    g = CognitiveGenome(
        sensors=sensors,
        bn_structure=BayesianNetworkStructure(edges),
    )
    return g


def blind_but_optimistic() -> CognitiveGenome:
    """Sees nothing relevant but has optimistic priors."""
    g = CognitiveGenome(
        sensors=frozenset({FoodAttribute.LUMINOSITY}),  # Not useful in most realities
        bn_structure=BayesianNetworkStructure(frozenset()),  # Models nothing!
        prior=CognitivePrior(global_mean=3.0, uncertainty_bonus=2.0),
    )
    return g


def overfitted_genome() -> CognitiveGenome:
    """Perceives everything, models everything — may overfit."""
    sensors = PERCEPTIBLE_ATTRIBUTES
    edges = frozenset(BNEdge(s, 'ENERGY') for s in sensors)
    g = CognitiveGenome(
        sensors=frozenset(sensors),
        bn_structure=BayesianNetworkStructure(edges),
        prior=CognitivePrior(pseudo_observations=0.01),  # Weak priors, learns fast
    )
    return g


GENOME_PRESETS = {
    'minimal': minimal_genome,
    'standard': standard_genome,
    'rich': rich_genome,
    'blind_optimist': blind_but_optimistic,
    'overfitted': overfitted_genome,
}
