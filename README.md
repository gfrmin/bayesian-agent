# Bayesian Agent Grid World

A simulation of an autonomous agent learning to navigate and survive in an environment using Bayesian inference and Thompson sampling.

## For Users

### What is this?

Watch an AI agent learn from experience in real-time! The agent starts with no knowledge about which foods are good or bad. Through trial and error, it builds up beliefs about food types using **Bayesian probability**, gradually learning to prefer high-energy foods and avoid toxic ones.

### Quick Start

```bash
# Install dependencies
pip install numpy

# Run the simulation
python main.py
```

### Controls

- **SPACE**: Pause/unpause the simulation
- **q**: Quit

### What you'll see

- **@**: The agent (you!)
- **●, ■, ▲**: Food items (circles, squares, triangles) in different colors
- **Belief table**: Shows what the agent has learned about each food type
  - Higher numbers = more energy
  - Negative numbers = toxic (drains energy)
- **Stats**: Energy level, position, steps taken, observations made

The agent starts with 10 energy and loses 0.1 energy per step. It must learn which foods to eat before running out of energy!

### Watch the Learning Process

At first, the agent explores randomly. As it gathers data, you'll see its beliefs converge toward the true energy values. Watch how it balances:
- **Exploitation**: Going for foods it knows are good
- **Exploration**: Trying unfamiliar foods to learn more

---

## Technical Deep Dive: Bayesian Cognition

### The Core Principle

The agent implements a minimal but complete Bayesian cognitive architecture. Rather than using fixed rules or deep learning, it maintains **probabilistic beliefs** about the world and updates them through experience using exact Bayesian inference.

### Belief Representation

The agent maintains a belief distribution over the expected energy value for each food type (shape, color) pair:

```
P(μ_energy | shape, color, observations)
```

This is represented as a **Normal distribution** characterized by:
- **μ**: mean (expected energy)
- **σ²**: variance (uncertainty)
- **n**: pseudo-observation count (effective sample size)

Initial state (prior):
```python
μ₀ = 0.0      # Neutral expectation
σ₀² = 10.0    # High uncertainty
n₀ = 0.1      # Weak prior (easily overridden)
```

### Bayesian Update Mechanism

When the agent consumes food and observes energy `x`, it updates its belief using **conjugate Bayesian inference**. The implementation (`bayesian_agent.py:40-77`) uses a Normal-Normal conjugate update:

**Prior**: μ ~ N(μ₀, σ₀²)
**Likelihood**: x ~ N(μ, σ²)
**Posterior**: μ ~ N(μ₁, σ₁²)

The update equations:

```
# Posterior variance (uncertainty decreases)
σ₁² = σ₀² / (1 + w/n)

# Posterior mean (weighted average of prior and observation)
μ₁ = (n·μ₀ + w·x) / (n + w)
```

where:
- `w` = observation weight (set to 1.0)
- `n` = prior pseudo-observation count
- `σ₀²` = prior variance

This update has several elegant properties:

1. **Automatic confidence weighting**: Early observations have large impact; later ones refine estimates
2. **Variance reduction**: σ₁² < σ₀², uncertainty decreases monotonically with data
3. **Online learning**: No need to store past observations; belief is a sufficient statistic
4. **Exact inference**: No approximation—this is the true Bayesian posterior

### Decision Making: Thompson Sampling

The agent uses **Thompson Sampling** (`bayesian_agent.py:110-138`) for the exploration-exploitation tradeoff. This is a probability-matching strategy that is both principled and efficient.

**Algorithm**:
```
For each visible food:
    1. Sample energy from posterior: ε ~ N(μ, σ²)
    2. Account for distance cost: value = ε - distance·cost
    3. Select food with highest sampled value
```

**Why Thompson Sampling?**

- **Optimality**: Provably optimal in the multi-armed bandit setting
- **Natural exploration**: High uncertainty → wider sampling → more exploration
- **Computational efficiency**: O(k) samples for k options vs. O(k log k) for UCB
- **Regret bounds**: Logarithmic regret O(log T) for T trials

**Contrast with UCB** (alternative mode):
```
value = μ + c·σ - distance·cost
```

Thompson Sampling is randomized (different decisions for same state), while UCB is deterministic. Thompson tends to explore more in this domain.

### Information Architecture

**State space**:
- Position: (x, y) ∈ ℤ² (discrete grid)
- Energy: E ∈ ℝ (continuous, death at E ≤ 0)
- Beliefs: {(shape, color) → N(μ, σ²)} (12 distributions)

**Observation space**:
- Visual: Set of (position, shape, color) tuples within radius
- Proprioceptive: Energy change after consumption

**Information flow**:
```
Environment → Observation → Belief Update → Decision → Action → Environment
     ↑                                                              |
     └──────────────────────────────────────────────────────────────┘
```

### Learning Dynamics

The agent exhibits several emergent behaviors:

1. **Initial random exploration**: High variance → Thompson samples are dispersed
2. **Preference formation**: Positive feedback → μ increases → higher selection probability
3. **Aversion learning**: Negative feedback → μ decreases → avoidance
4. **Uncertainty-driven exploration**: Untested foods have high σ → occasionally sampled
5. **Convergence**: As n → ∞, σ² → 0, behavior becomes exploitation-dominant

**Convergence rate**: After n observations of food type k:
```
σₙ² ≈ σ₀² / n
```

Standard error decreases as O(1/√n), typical for maximum likelihood estimation.

### Mathematical Foundation

The core assumption is that energy values are **normally distributed**:

```
energy ~ N(μ_true, σ_true²)
```

where (μ_true, σ_true²) are ground truth parameters defined in `config.py:22-40`.

Example distributions:
- Circle + Red: N(3.0, 1.0) - consistently good
- Square + Red: N(-1.0, 2.0) - risky, often toxic
- Triangle + Red: N(-2.0, 1.0) - reliably toxic

The agent doesn't know these true values—it must infer them from samples.

### Implementation Details

**Key files**:
- `bayesian_agent.py`: Core Bayesian inference engine
- `environment.py`: World simulator with stochastic food generation
- `config.py`: Ground truth distributions and hyperparameters
- `main.py`: Simulation loop with curses rendering

**Critical functions**:
- `BayesianAgent.update_belief()`: Bayesian posterior update
- `BayesianAgent.sample_energy()`: Thompson sampling draw
- `BayesianAgent.select_target_food()`: Decision policy
- `GridWorld.spawn_food()`: Stochastic environment dynamics

**Hyperparameters**:
- `MOVEMENT_COST = 0.1`: Energy cost per step (encourages efficiency)
- `PRIOR_BELIEF['variance'] = 10.0`: Initial uncertainty magnitude
- `THOMPSON_SAMPLING = True`: Toggle Thompson vs. UCB
- `UCB_C = 2.0`: Exploration bonus for UCB mode

### Why This Matters

This architecture demonstrates that **probabilistic inference** provides a principled framework for learning and decision-making under uncertainty. Unlike:

- **Reinforcement Learning**: No need for reward engineering, credit assignment, or neural networks
- **Rule-based systems**: Handles uncertainty naturally, no brittle thresholds
- **Pure exploration**: Automatically balances exploration/exploitation

The agent's behavior emerges from first principles of probability theory. This is a minimal example of **Bayesian brain hypothesis** in action—the idea that cognition is fundamentally Bayesian inference.

---

## For Developers

### Project Structure

```
bayesian-agent/
├── main.py              # Simulation loop + curses UI
├── bayesian_agent.py    # Agent with Bayesian belief updating
├── environment.py       # Grid world + food spawning
├── config.py           # Parameters + ground truth distributions
└── pyproject.toml      # Dependencies
```

### Architecture

**Separation of concerns**:
- **Agent**: Internal beliefs and decision-making (no environment knowledge)
- **Environment**: World state and dynamics (no agent knowledge)
- **Main**: Orchestration and rendering

**Agent interface**:
```python
agent.select_target_food(available_foods) → position
agent.move(position) → None
agent.consume(shape, color, energy) → None  # Triggers belief update
agent.get_belief_summary() → Dict[Tuple, Dict]
```

**Environment interface**:
```python
env.update() → None  # Spawn food probabilistically
env.get_nearby_foods(position, radius) → List[Food]
env.consume_food(position) → Optional[Tuple[shape, color, energy]]
```

### Extending the Simulation

**Add new food types**:
1. Edit `SHAPES` or `COLORS` in `config.py`
2. Add distribution to `ENERGY_DISTRIBUTIONS`
3. Agent automatically handles new types (general Bayesian update)

**Try different priors**:
```python
PRIOR_BELIEF = {
    "mean": 2.0,         # Optimistic prior
    "variance": 1.0,     # Low uncertainty (strong prior)
    "pseudo_observations": 10.0  # Hard to override
}
```

**Switch decision strategies**:
```python
THOMPSON_SAMPLING = False  # Use UCB instead
UCB_C = 3.0               # More exploration
```

**Modify belief update**:
- Current: Assumes known variance, updates mean only
- Extension: Use Normal-Gamma conjugate prior to learn variance too
- See: `bayesian_agent.py:40-77`

### Key Algorithms

**Bayesian update** (`bayesian_agent.py:40-77`):
```python
def update_belief(self, shape, color, observed_energy):
    n = belief["n"]
    prior_mean = belief["mean"]

    # Weighted average
    new_mean = (n * prior_mean + observed_energy) / (n + 1)

    # Shrink variance
    new_variance = prior_variance / (1 + 1/n)

    belief["mean"] = new_mean
    belief["variance"] = new_variance
    belief["n"] = n + 1
```

**Thompson sampling** (`bayesian_agent.py:123-138`):
```python
def select_target_food(self, available_foods):
    best_value = -inf
    for position, shape, color in available_foods:
        # Sample from posterior
        sampled = random.normal(mean, std)
        value = sampled - distance * MOVEMENT_COST
        if value > best_value:
            best_position = position
    return best_position
```

### Testing Ideas

1. **Visualize learning curves**: Track belief convergence vs. true values
2. **Compare strategies**: Thompson vs. UCB vs. greedy vs. random
3. **Measure regret**: Cumulative energy vs. oracle with perfect knowledge
4. **Parameter sensitivity**: How does prior strength affect learning speed?
5. **Non-stationary environments**: Change distributions mid-simulation

### Performance Notes

- **Rendering**: Curses-based, ~10 FPS on typical terminal
- **Belief updates**: O(1) per observation (no history needed)
- **Decision making**: O(k) for k visible foods
- **Memory**: O(n) for n food types, O(m) for m observations logged

### Common Modifications

**Make agent learn faster**:
```python
PRIOR_BELIEF['pseudo_observations'] = 0.01  # Weaker prior
```

**Make environment harder**:
```python
MOVEMENT_COST = 0.5  # Higher cost
INITIAL_ENERGY = 5.0  # Less starting energy
```

**Add obstacles/walls**: Modify `GridWorld` and `get_next_move()` for pathfinding

**Multi-agent**: Instantiate multiple `BayesianAgent` objects with shared environment

---

## Roadmap

### Near-term (v0.2)
- [ ] Add variance learning (Normal-Gamma conjugate prior)
- [ ] Implement greedy and random baselines for comparison
- [ ] Add logging to CSV for post-hoc analysis
- [ ] Visualize belief evolution as time series plots

### Medium-term (v0.3)
- [ ] Non-stationary environment (distributions drift over time)
- [ ] Spatial correlations (food clusters)
- [ ] Multi-agent scenarios (competition/cooperation)
- [ ] Add obstacles and pathfinding (A* algorithm)

### Long-term (v1.0)
- [ ] Hierarchical beliefs (learn feature correlations)
- [ ] Active learning (agent requests specific observations)
- [ ] Causal reasoning (interventions vs. observations)
- [ ] Model-based planning (forward simulation)
- [ ] Transfer learning (generalize to new environments)

### Research Extensions
- [ ] Compare with RL baselines (Q-learning, DQN)
- [ ] Analyze sample efficiency vs. neural approaches
- [ ] Study emergent exploration strategies
- [ ] Characterize regret bounds empirically
- [ ] Implement Bayesian optimization for hyperparameter tuning

---

## References

### Bayesian Inference
- Gelman et al. (2013). *Bayesian Data Analysis*. CRC Press.
- Murphy (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

### Thompson Sampling
- Russo et al. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*.
- Agrawal & Goyal (2012). "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." *COLT*.

### Bayesian Brain Hypothesis
- Friston (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.
- Doya et al. (2007). *Bayesian Brain: Probabilistic Approaches to Neural Coding*. MIT Press.

---

## License

MIT License - See code for details.

## Contributing

Contributions welcome! Key areas:
- Algorithmic improvements (better inference, planning)
- Visualization enhancements (plots, metrics)
- Performance optimization
- New environment dynamics
- Documentation and examples

Open an issue or submit a PR.