"""Bayesian agent with probabilistic belief updating and action selection."""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
from config import (
    INITIAL_ENERGY, MOVEMENT_COST, STARTING_POSITION,
    SHAPES, COLORS, PRIOR_BELIEF, THOMPSON_SAMPLING, UCB_C
)


class BayesianAgent:
    """Agent that maintains Bayesian beliefs over food energy values."""

    def __init__(self):
        self.position = STARTING_POSITION
        self.energy = INITIAL_ENERGY

        # Beliefs: P(energy | shape, color)
        # Using Normal distribution with mean and variance
        # Store (mean, variance, n_observations) for each (shape, color)
        self.beliefs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._initialize_beliefs()

        # History
        self.observations: List[Dict] = []
        self.energy_history: List[Dict] = []
        self.total_steps = 0

    def _initialize_beliefs(self):
        """Initialize prior beliefs for all food types."""
        for shape in SHAPES:
            for color in COLORS:
                self.beliefs[(shape, color)] = {
                    "mean": PRIOR_BELIEF["mean"],
                    "variance": PRIOR_BELIEF["variance"],
                    "n": PRIOR_BELIEF["pseudo_observations"]
                }

    def update_belief(self, shape: str, color: str, observed_energy: float):
        """
        Update belief using Bayesian inference (conjugate Normal-Normal update).

        For Normal likelihood with known variance σ²:
        Prior: μ ~ N(μ₀, σ₀²)
        Likelihood: x ~ N(μ, σ²)
        Posterior: μ ~ N(μ₁, σ₁²)

        Where:
        σ₁² = 1 / (1/σ₀² + n/σ²)
        μ₁ = σ₁² * (μ₀/σ₀² + Σx/σ²)
        """
        key = (shape, color)
        belief = self.beliefs[key]

        # Prior parameters
        prior_mean = belief["mean"]
        prior_variance = belief["variance"]
        n = belief["n"]

        # For simplicity, assume observation variance equals prior variance
        # This makes the update a weighted average
        observation_weight = 1.0
        total_weight = n + observation_weight

        # Update mean (weighted average)
        new_mean = (n * prior_mean + observation_weight * observed_energy) / total_weight

        # Update variance (decreases with more observations)
        new_variance = prior_variance / (1 + observation_weight / n)

        # Update belief
        self.beliefs[key] = {
            "mean": new_mean,
            "variance": new_variance,
            "n": n + observation_weight
        }

        # Record observation
        self.observations.append({
            "shape": shape,
            "color": color,
            "energy": observed_energy,
            "step": self.total_steps
        })

    def get_expected_energy(self, shape: str, color: str) -> float:
        """Get expected energy for a food type."""
        return self.beliefs[(shape, color)]["mean"]

    def get_uncertainty(self, shape: str, color: str) -> float:
        """Get uncertainty (standard deviation) for a food type."""
        return np.sqrt(self.beliefs[(shape, color)]["variance"])

    def sample_energy(self, shape: str, color: str) -> float:
        """Sample energy from posterior distribution (Thompson Sampling)."""
        belief = self.beliefs[(shape, color)]
        mean = belief["mean"]
        std = np.sqrt(belief["variance"])
        return np.random.normal(mean, std)

    def get_ucb(self, shape: str, color: str) -> float:
        """
        Calculate Upper Confidence Bound.
        UCB = mean + c * sqrt(variance)
        """
        belief = self.beliefs[(shape, color)]
        return belief["mean"] + UCB_C * np.sqrt(belief["variance"])

    def select_target_food(self, available_foods: List[Tuple[Tuple[int, int], str, str]]) -> Optional[Tuple[int, int]]:
        """
        Select which food to target based on beliefs.

        Args:
            available_foods: List of (position, shape, color) tuples

        Returns:
            Target position or None
        """
        if not available_foods:
            return None

        if THOMPSON_SAMPLING:
            # Thompson Sampling: sample from posteriors and pick best
            best_value = float('-inf')
            best_position = None

            for position, shape, color in available_foods:
                sampled_energy = self.sample_energy(shape, color)
                # Account for distance cost
                distance = abs(position[0] - self.position[0]) + abs(position[1] - self.position[1])
                value = sampled_energy - distance * MOVEMENT_COST

                if value > best_value:
                    best_value = value
                    best_position = position

            return best_position
        else:
            # UCB: pick highest upper confidence bound
            best_ucb = float('-inf')
            best_position = None

            for position, shape, color in available_foods:
                ucb = self.get_ucb(shape, color)
                distance = abs(position[0] - self.position[0]) + abs(position[1] - self.position[1])
                value = ucb - distance * MOVEMENT_COST

                if value > best_ucb:
                    best_ucb = value
                    best_position = position

            return best_position

    def get_next_move(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get next position moving toward target (Manhattan distance).

        Returns:
            Next position (x, y)
        """
        x, y = self.position
        tx, ty = target

        # Move one step toward target
        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        elif y < ty:
            y += 1
        elif y > ty:
            y -= 1

        return (x, y)

    def move(self, new_position: Tuple[int, int]):
        """Move to new position and pay movement cost."""
        energy_before = self.energy
        self.position = new_position
        self.energy -= MOVEMENT_COST
        self.total_steps += 1

        # Log energy change (step will be index+1 when displayed)
        self.energy_history.append({
            "energy_before": energy_before,
            "energy_after": self.energy,
            "delta": -MOVEMENT_COST,
            "reason": "movement"
        })

    def consume(self, shape: str, color: str, energy: float):
        """Consume food and update beliefs."""
        energy_before = self.energy
        self.energy += energy
        self.update_belief(shape, color, energy)

        # Log energy change (step will be index+1 when displayed)
        self.energy_history.append({
            "energy_before": energy_before,
            "energy_after": self.energy,
            "delta": energy,
            "reason": f"ate {shape} {color}"
        })

    def get_state(self) -> Dict:
        """Get current agent state."""
        return {
            "position": self.position,
            "energy": self.energy,
            "total_observations": len(self.observations),
            "total_steps": self.total_steps
        }

    def get_belief_summary(self) -> Dict:
        """Get summary of current beliefs."""
        summary = {}
        for key, belief in self.beliefs.items():
            shape, color = key
            summary[key] = {
                "mean": belief["mean"],
                "std": np.sqrt(belief["variance"]),
                "n": belief["n"]
            }
        return summary