# Evolutionary Bayesian Agents

A simulation of evolving populations of Bayesian agents learning to forage in environments with hidden causal structure.

## Core Insight

Two timescales of adaptation:

| Timescale | Mechanism | What Adapts |
|-----------|-----------|-------------|
| **Ontogenetic** (within lifetime) | Bayesian inference | Beliefs (probability distributions) |
| **Phylogenetic** (across generations) | Natural selection | Cognitive architecture (genome) |

The genome specifies *how* to learn; learning fills in *what* is believed.

## Quick Start

```bash
# Interactive mode — curses visualization of population foraging
uv run main.py

# Batch mode — text-only evolutionary run
uv run main.py --batch

# Options
uv run main.py --reality interaction --population 50 --generations 100
uv run main.py --batch --reality hierarchical
```

### Controls (interactive mode)

- **SPACE** — pause / resume
- **q** — quit
- **+/-** — adjust simulation speed

### What you'll see

- **Grid**: Food items rendered as coloured shapes (●, ■, ▲, ⬡) and agents as `@`
- **Stats**: Generation, tick, alive/total agents, mean/max fitness, survival rate
- **Beliefs**: The best agent's current learned beliefs about food types

## Available Realities

| Reality | Description | What Agents Should Learn |
|---------|-------------|--------------------------|
| `simple` | Independent shape + color effects | Shape and color each matter |
| `interaction` | Shape-color combinations | Green circles are super-nutritious |
| `hierarchical` | Multi-level dependencies | Temperature affects texture safety |
| `temporal` | Time-varying effects | Hidden cycles, real-world time |
| `hidden` | Hidden variables | Position matters, but can't be seen |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  REALITY (defined by the "god" — can be arbitrarily complex)    │
│  • True causal structure of food → energy                       │
│  • May include hidden variables agents cannot perceive          │
└─────────────────────────────────────────────────────────────────┘
                              │ observations (filtered by genome)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT (genome specifies structure, learning fills parameters)  │
│  • Sensors: which attributes can be perceived                   │
│  • BN structure: which variables are modelled as relevant       │
│  • Priors: initial beliefs before learning                      │
└─────────────────────────────────────────────────────────────────┘
                              │ actions → outcomes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SELECTION                                                      │
│  • Fitness = net energy (gained − spent)                        │
│  • Fitter agents reproduce; their genomes propagate             │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
bayesian-agent/
├── main.py              # Entry point: interactive (curses) or batch mode
├── reality.py           # Ground truth causal structure of the world
├── cognitive_genome.py  # Heritable cognitive architecture specification
├── bayesian_agent.py    # Agents that learn via Bayesian inference + populations
├── simulation.py        # Evolutionary simulation driver
├── visualization.py     # Curses-based population renderer
└── pyproject.toml       # Dependencies (just numpy)
```

## Key Concepts

### The Genome

Encodes:
1. **Sensors** — which attributes the agent can perceive (e.g., SHAPE, COLOR, TEMPERATURE)
2. **BN Structure** — which perceived attributes are modelled as relevant to energy
3. **Priors** — initial belief parameters (mean, variance, pseudo-observations)

The genome does NOT encode actual beliefs — those are learned within a lifetime.

### Bayesian Learning

Agents update beliefs using conjugate Normal-Normal inference:

```
P(μ | observations) ~ Normal(posterior_mean, posterior_variance)
```

Each configuration of relevant variables gets its own belief distribution.

### Thompson Sampling

Agents use Thompson sampling for exploration-exploitation: sample from the posterior, pick the food with highest sampled value minus travel cost.

### Evolution

Across generations:
1. Agents with higher fitness reproduce
2. Offspring inherit mutated cognitive architectures
3. Architectures that enable better learning propagate

## References

- Friston (2010). The free-energy principle: a unified brain theory?
- Baldwin (1896). A New Factor in Evolution
- Hinton & Nowlan (1987). How Learning Can Guide Evolution
- Russo et al. (2018). A Tutorial on Thompson Sampling

## License

MIT
