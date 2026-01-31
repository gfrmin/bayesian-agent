"""
Reality: The true causal structure of the world.

This module defines GROUND TRUTH — the actual rules governing how
observations and outcomes relate. Agents never see this code; they
must infer (imperfect) approximations through experience.

Key Design Principles:
---------------------
1. Reality can be arbitrarily complex
2. Reality can include variables agents cannot perceive
3. The structure can be changed to pose different "problems"
4. There's no requirement that reality be "fair" or learnable

The Gap Between Map and Territory:
---------------------------------
- TERRITORY (this module): True causal structure
- MAP (agent's BN): Agent's theory of the structure

Evolution discovers which maps work well for which territories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from enum import Enum, auto
import random
import math
from datetime import datetime


class FoodAttribute(Enum):
    """All possible food attributes that could exist in reality."""
    # Perceptible attributes (agents could potentially sense these)
    SHAPE = auto()          # circle, square, triangle, etc.
    COLOR = auto()          # red, green, blue, etc.
    SIZE = auto()           # small, medium, large
    TEXTURE = auto()        # smooth, rough, spiky
    LUMINOSITY = auto()     # dull, normal, glowing
    SOUND = auto()          # silent, humming, clicking
    SMELL = auto()          # none, sweet, acrid
    TEMPERATURE = auto()    # cold, warm, hot
    
    # Hidden attributes (agents cannot perceive these directly)
    ORIGIN_QUADRANT = auto()     # where in the world it spawned
    TIME_OF_SPAWN = auto()       # when it appeared
    SPAWN_CYCLE_PHASE = auto()   # phase of some hidden cycle
    MOLECULAR_STRUCTURE = auto() # hidden "chemistry"
    
    # Meta-attributes (external to simulation)
    REAL_WORLD_HOUR = auto()     # actual hour in user's timezone
    REAL_WORLD_WEEKDAY = auto()  # is it a weekday?


# Which attributes are POSSIBLY perceptible (vs always hidden)
PERCEPTIBLE_ATTRIBUTES = {
    FoodAttribute.SHAPE,
    FoodAttribute.COLOR,
    FoodAttribute.SIZE,
    FoodAttribute.TEXTURE,
    FoodAttribute.LUMINOSITY,
    FoodAttribute.SOUND,
    FoodAttribute.SMELL,
    FoodAttribute.TEMPERATURE,
}


@dataclass
class FoodItem:
    """A food item with all its attributes."""
    attributes: Dict[FoodAttribute, Any]
    position: Tuple[int, int] = (0, 0)
    
    def get(self, attr: FoodAttribute, default=None) -> Any:
        return self.attributes.get(attr, default)
    
    def get_perceptible(self, sensor_suite: Set[FoodAttribute]) -> Dict[FoodAttribute, Any]:
        """Return only attributes this agent can perceive."""
        return {
            attr: val for attr, val in self.attributes.items()
            if attr in sensor_suite and attr in PERCEPTIBLE_ATTRIBUTES
        }


@dataclass 
class CausalFactor:
    """
    A factor that contributes to the true energy value of food.
    
    The actual energy of food is: sum of all CausalFactor contributions
    """
    name: str
    condition: Callable[[FoodItem, Dict[str, Any]], bool]  # When does this apply?
    effect: Callable[[FoodItem, Dict[str, Any]], float]     # What's the contribution?
    description: str = ""


class Reality:
    """
    The ground truth causal structure of the world.
    
    This is the TERRITORY that agents' maps try to approximate.
    We (the gods) can make this as simple or as devilishly complex
    as we like.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (20, 20),
                 timezone_offset: int = 0):  # Hours from UTC
        self.grid_size = grid_size
        self.timezone_offset = timezone_offset
        self.causal_factors: List[CausalFactor] = []
        self.time_step = 0
        
        # Hidden state that affects reality but agents can't see
        self.hidden_cycle_phase = 0.0  # oscillates 0 to 2π
        self.hidden_cycle_period = 100  # time steps per cycle
        
        # Attribute value sets
        self.attribute_values = {
            FoodAttribute.SHAPE: ['circle', 'square', 'triangle', 'hexagon'],
            FoodAttribute.COLOR: ['red', 'green', 'blue', 'yellow'],
            FoodAttribute.SIZE: ['small', 'medium', 'large'],
            FoodAttribute.TEXTURE: ['smooth', 'rough', 'spiky'],
            FoodAttribute.LUMINOSITY: ['dull', 'normal', 'glowing'],
            FoodAttribute.SOUND: ['silent', 'humming', 'clicking'],
            FoodAttribute.SMELL: ['none', 'sweet', 'acrid'],
            FoodAttribute.TEMPERATURE: ['cold', 'warm', 'hot'],
        }
    
    def add_causal_factor(self, factor: CausalFactor) -> 'Reality':
        """Fluent interface for adding causal factors."""
        self.causal_factors.append(factor)
        return self
    
    def get_real_world_context(self) -> Dict[str, Any]:
        """Get external context (things outside the simulation)."""
        now = datetime.utcnow()
        local_hour = (now.hour + self.timezone_offset) % 24
        weekday = now.weekday()
        return {
            'hour': local_hour,
            'minute': now.minute,
            'weekday': weekday,  # 0=Monday, 6=Sunday
            'is_weekend': weekday >= 5,
            'is_night': local_hour < 6 or local_hour >= 22,
            'is_working_hours': 9 <= local_hour < 17 and weekday < 5,
        }
    
    def get_simulation_context(self) -> Dict[str, Any]:
        """Get internal simulation context."""
        return {
            'time_step': self.time_step,
            'cycle_phase': self.hidden_cycle_phase,
            'cycle_sin': math.sin(self.hidden_cycle_phase),
            'cycle_cos': math.cos(self.hidden_cycle_phase),
        }
    
    def get_full_context(self) -> Dict[str, Any]:
        """Combine all context for causal factor evaluation."""
        ctx = self.get_simulation_context()
        ctx.update(self.get_real_world_context())
        return ctx
    
    def compute_true_energy(self, food: FoodItem) -> float:
        """
        Compute the ACTUAL energy value of a food item.
        
        This is ground truth — what really happens when you eat it.
        The agent's prediction may be wildly off.
        """
        context = self.get_full_context()
        
        total_energy = 0.0
        for factor in self.causal_factors:
            if factor.condition(food, context):
                contribution = factor.effect(food, context)
                total_energy += contribution
        
        return total_energy
    
    def spawn_food(self, position: Optional[Tuple[int, int]] = None) -> FoodItem:
        """
        Generate a new food item with random attributes.
        """
        if position is None:
            position = (
                random.randint(0, self.grid_size[0] - 1),
                random.randint(0, self.grid_size[1] - 1)
            )
        
        context = self.get_full_context()
        
        attributes = {}
        for attr, values in self.attribute_values.items():
            attributes[attr] = random.choice(values)
        
        # Add hidden attributes
        attributes[FoodAttribute.ORIGIN_QUADRANT] = (
            'NW' if position[0] < self.grid_size[0]//2 and position[1] < self.grid_size[1]//2 else
            'NE' if position[0] >= self.grid_size[0]//2 and position[1] < self.grid_size[1]//2 else
            'SW' if position[0] < self.grid_size[0]//2 else 'SE'
        )
        attributes[FoodAttribute.TIME_OF_SPAWN] = self.time_step
        attributes[FoodAttribute.SPAWN_CYCLE_PHASE] = self.hidden_cycle_phase
        attributes[FoodAttribute.REAL_WORLD_HOUR] = context['hour']
        
        return FoodItem(attributes=attributes, position=position)
    
    def step(self):
        """Advance simulation time."""
        self.time_step += 1
        self.hidden_cycle_phase = (2 * math.pi * self.time_step / self.hidden_cycle_period) % (2 * math.pi)


# =============================================================================
# PRESET REALITIES: Different "problems" to solve
# =============================================================================

def create_simple_reality() -> Reality:
    """
    Simple reality: just shape and color matter, independently.
    """
    reality = Reality()
    
    # Base nutrition (ensures most food is at least slightly positive)
    reality.add_causal_factor(CausalFactor(
        name="base_nutrition",
        condition=lambda f, c: True,
        effect=lambda f, c: 2.0,  # Everything has 2 base energy
    ))
    
    # Base energy by shape (relative to base)
    shape_effects = {'circle': 2.0, 'square': 0.0, 'triangle': -1.5, 'hexagon': 0.5}
    for shape, effect in shape_effects.items():
        reality.add_causal_factor(CausalFactor(
            name=f"{shape}_effect",
            condition=lambda f, c, s=shape: f.get(FoodAttribute.SHAPE) == s,
            effect=lambda f, c, e=effect: e,
        ))
    
    # Modifier by color
    color_effects = {'green': 1.0, 'red': -0.5, 'blue': 0.3, 'yellow': 0.0}
    for color, effect in color_effects.items():
        reality.add_causal_factor(CausalFactor(
            name=f"{color}_effect",
            condition=lambda f, c, col=color: f.get(FoodAttribute.COLOR) == col,
            effect=lambda f, c, e=effect: e,
        ))
    
    return reality


def create_interaction_reality() -> Reality:
    """
    Reality with interactions: shape-color combinations matter.
    """
    reality = Reality()
    
    # Base nutrition
    reality.add_causal_factor(CausalFactor(
        name="base_nutrition",
        condition=lambda f, c: True,
        effect=lambda f, c: 1.5,
    ))
    
    # Weak base effects
    reality.add_causal_factor(CausalFactor(
        name="circle_base",
        condition=lambda f, c: f.get(FoodAttribute.SHAPE) == 'circle',
        effect=lambda f, c: 0.5,
    ))
    reality.add_causal_factor(CausalFactor(
        name="green_base",
        condition=lambda f, c: f.get(FoodAttribute.COLOR) == 'green',
        effect=lambda f, c: 0.3,
    ))
    
    # Strong interaction effects (the real signal)
    reality.add_causal_factor(CausalFactor(
        name="green_circle_synergy",
        condition=lambda f, c: (f.get(FoodAttribute.SHAPE) == 'circle' and 
                                f.get(FoodAttribute.COLOR) == 'green'),
        effect=lambda f, c: 3.0,  # Big bonus for the combo
    ))
    reality.add_causal_factor(CausalFactor(
        name="red_triangle_danger",
        condition=lambda f, c: (f.get(FoodAttribute.SHAPE) == 'triangle' and 
                                f.get(FoodAttribute.COLOR) == 'red'),
        effect=lambda f, c: -4.0,  # Still dangerous but not instant death
    ))
    
    return reality
    
    return reality


def create_hidden_variable_reality() -> Reality:
    """
    Reality with hidden variables: observable attributes are incomplete.
    """
    reality = Reality()
    
    # The REAL rule: it's all about where food spawns (hidden)
    reality.add_causal_factor(CausalFactor(
        name="northwest_toxic",
        condition=lambda f, c: f.get(FoodAttribute.ORIGIN_QUADRANT) == 'NW',
        effect=lambda f, c: -5.0,
    ))
    reality.add_causal_factor(CausalFactor(
        name="southeast_nutritious",
        condition=lambda f, c: f.get(FoodAttribute.ORIGIN_QUADRANT) == 'SE',
        effect=lambda f, c: 4.0,
    ))
    
    # Base nutrition
    reality.add_causal_factor(CausalFactor(
        name="base",
        condition=lambda f, c: True,
        effect=lambda f, c: 1.0,
    ))
    
    return reality


def create_temporal_reality() -> Reality:
    """
    Reality with temporal structure: things change over time.
    """
    reality = Reality()
    
    # Internal cycle
    reality.add_causal_factor(CausalFactor(
        name="circle_cycle",
        condition=lambda f, c: f.get(FoodAttribute.SHAPE) == 'circle',
        effect=lambda f, c: 3.0 * c['cycle_sin'],
    ))
    
    # Real-world time effects
    reality.add_causal_factor(CausalFactor(
        name="night_bonus",
        condition=lambda f, c: c['is_night'],
        effect=lambda f, c: 2.0,
    ))
    
    reality.add_causal_factor(CausalFactor(
        name="base",
        condition=lambda f, c: True,
        effect=lambda f, c: 2.0,
    ))
    
    return reality


def create_hierarchical_reality() -> Reality:
    """
    Reality with hierarchical causal structure.
    """
    reality = Reality()
    
    # Base color effects
    color_effects = {'red': -1.0, 'green': 2.0, 'blue': 0.5, 'yellow': 1.0}
    for color, effect in color_effects.items():
        reality.add_causal_factor(CausalFactor(
            name=f"{color}_base",
            condition=lambda f, c, col=color: f.get(FoodAttribute.COLOR) == col,
            effect=lambda f, c, eff=effect: eff,
        ))
    
    # Size multiplier
    size_mult = {'small': -0.5, 'medium': 0.0, 'large': 1.0}
    for size, mult in size_mult.items():
        reality.add_causal_factor(CausalFactor(
            name=f"{size}_mult",
            condition=lambda f, c, sz=size: f.get(FoodAttribute.SIZE) == sz,
            effect=lambda f, c, m=mult: m,
        ))
    
    # Temperature-texture interaction
    reality.add_causal_factor(CausalFactor(
        name="hot_spiky_danger",
        condition=lambda f, c: (f.get(FoodAttribute.TEMPERATURE) == 'hot' and
                                f.get(FoodAttribute.TEXTURE) == 'spiky'),
        effect=lambda f, c: -6.0,
    ))
    reality.add_causal_factor(CausalFactor(
        name="cold_smooth_safe",
        condition=lambda f, c: (f.get(FoodAttribute.TEMPERATURE) == 'cold' and
                                f.get(FoodAttribute.TEXTURE) == 'smooth'),
        effect=lambda f, c: 3.0,
    ))
    
    return reality


REALITY_PRESETS = {
    'simple': create_simple_reality,
    'interaction': create_interaction_reality,
    'hidden': create_hidden_variable_reality,
    'temporal': create_temporal_reality,
    'hierarchical': create_hierarchical_reality,
}
