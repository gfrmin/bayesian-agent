"""Grid world environment with probabilistic food spawning."""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import (
    GRID_WIDTH, GRID_HEIGHT, MAX_FOOD_ITEMS, FOOD_SPAWN_RATE,
    SHAPES, COLORS, ENERGY_DISTRIBUTIONS
)


class Food:
    """Represents a food item with visual features and energy."""

    def __init__(self, position: Tuple[int, int], shape: str, color: str):
        self.position = position
        self.shape = shape
        self.color = color
        # Sample energy from true distribution
        mean, std = ENERGY_DISTRIBUTIONS[(shape, color)]
        self.energy = np.random.normal(mean, std)

    def __repr__(self):
        return f"Food({self.shape}, {self.color}, E={self.energy:.1f})"


class GridWorld:
    """2D grid world with food items."""

    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.foods: List[Food] = []
        self.eaten_foods: List[Dict] = []  # History of consumed foods

    def spawn_food(self) -> Optional[Food]:
        """Attempt to spawn a new food item."""
        if len(self.foods) >= MAX_FOOD_ITEMS:
            return None

        if random.random() > FOOD_SPAWN_RATE:
            return None

        # Find empty position
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            position = (x, y)

            # Check if position is occupied
            if not any(f.position == position for f in self.foods):
                shape = random.choice(SHAPES)
                color = random.choice(COLORS)
                food = Food(position, shape, color)
                self.foods.append(food)
                return food

        return None

    def get_food_at(self, position: Tuple[int, int]) -> Optional[Food]:
        """Get food at given position."""
        for food in self.foods:
            if food.position == position:
                return food
        return None

    def consume_food(self, position: Tuple[int, int]) -> Optional[Tuple[str, str, float]]:
        """
        Consume food at position.
        Returns (shape, color, energy) if food exists, None otherwise.
        """
        food = self.get_food_at(position)
        if food:
            self.foods.remove(food)
            observation = (food.shape, food.color, food.energy)

            # Record in history
            self.eaten_foods.append({
                "shape": food.shape,
                "color": food.color,
                "energy": food.energy,
                "position": food.position
            })

            return observation
        return None

    def get_nearby_foods(self, position: Tuple[int, int], radius: int = 10) -> List[Food]:
        """Get foods within radius of position."""
        x, y = position
        nearby = []
        for food in self.foods:
            fx, fy = food.position
            distance = abs(fx - x) + abs(fy - y)  # Manhattan distance
            if distance <= radius:
                nearby.append(food)
        return nearby

    def update(self):
        """Update environment state (spawn new food)."""
        self.spawn_food()

    def get_state(self) -> Dict:
        """Get current environment state."""
        return {
            "foods": [(f.position, f.shape, f.color) for f in self.foods],
            "num_foods": len(self.foods),
            "total_consumed": len(self.eaten_foods)
        }