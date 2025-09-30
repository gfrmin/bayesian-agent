"""Configuration parameters for Bayesian Agent Grid World."""

# Grid dimensions
GRID_WIDTH = 30
GRID_HEIGHT = 15

# Agent parameters
INITIAL_ENERGY = 10.0
MOVEMENT_COST = 0.1
STARTING_POSITION = (GRID_WIDTH // 2, GRID_HEIGHT // 2)

# Food spawning
MAX_FOOD_ITEMS = 8
FOOD_SPAWN_RATE = 0.5  # Probability per tick to spawn new food

# Food features
SHAPES = ["●", "■", "▲"]  # circle, square, triangle
COLORS = ["red", "green", "yellow", "blue"]

# True underlying energy distributions P(energy | shape, color)
# Format: (mean, std_dev) for each (shape, color) combination
ENERGY_DISTRIBUTIONS = {
    # Circle: generally positive, color-dependent
    ("●", "red"): (3.0, 1.0),      # High energy
    ("●", "green"): (2.0, 1.5),    # Medium-high energy
    ("●", "yellow"): (1.0, 1.0),   # Medium energy
    ("●", "blue"): (0.5, 1.5),     # Low-medium energy

    # Square: mixed, high variance
    ("■", "red"): (-1.0, 2.0),     # Risky, could be negative
    ("■", "green"): (2.5, 2.0),    # Risky, could be high
    ("■", "yellow"): (0.0, 1.5),   # Neutral
    ("■", "blue"): (1.5, 1.0),     # Moderate positive

    # Triangle: generally negative, color matters
    ("▲", "red"): (-2.0, 1.0),     # Negative
    ("▲", "green"): (-0.5, 1.5),   # Slightly negative
    ("▲", "yellow"): (-1.5, 1.0),  # Negative
    ("▲", "blue"): (0.5, 2.0),     # Exception: sometimes positive
}

# Agent's prior beliefs (initial uncertainty)
# Using conjugate Normal-Gamma prior: (prior_mean, prior_precision, alpha, beta)
# For simplicity, we use Normal with known variance and update the mean
PRIOR_BELIEF = {
    "mean": 0.0,           # Prior mean for all food types
    "variance": 10.0,      # High initial uncertainty
    "pseudo_observations": 0.1  # Weak prior (easy to update)
}

# Action selection parameters
THOMPSON_SAMPLING = True
UCB_C = 2.0  # Exploration constant for UCB (if not using Thompson Sampling)

# Display settings
SHOW_BELIEFS = True
UPDATE_DELAY = 0.1  # Seconds between updates

# Terminal color codes
COLOR_CODES = {
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
}