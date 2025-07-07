# Preference Falsification Simulation

A multi-agent simulation where LLM agents balance public reputation with private family needs, creating natural preference falsification dynamics.


## Quick Start

```bash
# 1. Configure the simulation (interactive setup)
python setup_config.py

# 2. Run the simulation
python pf_main_v2.py

# 3. Analyze results
python pf_analyzer.py pf_results/[your-results-file].json
```

## Overview

This simulation explores how agents behave when faced with competing incentives:
- **Public**: Maintain high reputation through generous contributions to community pool
- **Private**: Feed their family by keeping enough tokens for themselves

The tension between these goals creates preference falsification - agents may publicly support high contributions while privately needing to keep tokens for family survival.

## Key Features

### Core Mechanics
- **Token System**: Agents receive 10 tokens per round (7 if low reputation)
- **Public Pool**: Contributions multiplied by 1.5x and shared equally
- **Reputation**: Average contribution over last 5 rounds, visible to all
- **Family Needs**: Private requirement of 4 tokens/round (6 during crisis)

### Crisis Event
- Rounds 15-17: Community multiplier increases to 2x
- Family needs jump to 6 tokens
- Intensifies the public-private tension

### Communication
- **Public Statements**: Agents can make statements each round
- **Group Chat**: Community discussions
- All communication is public (family status remains private)

## Installation & Usage

### Prerequisites
- Python 3.8+
- API key for one of: OpenAI, Azure OpenAI, OpenRouter, or KlusterAI

### Configuration System

The simulation uses a centralized `config.py` file for all settings. You can:

1. **Use Interactive Setup** (Recommended for first-time users):
   ```bash
   python setup_config.py
   ```

2. **Edit config.py Directly**:
   ```python
   # config.py
   API_PROVIDER = 'openai'
   NUM_AGENTS = 12
   NUM_ROUNDS = 20
   CRISIS_START_ROUND = 15
   # ... etc
   ```

3. **Use Environment Variables** (for API keys):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export AZURE_API_KEY="your-azure-key"
   ```

### Running the Simulation

```bash
# Run with config.py settings
python pf_main_v2.py

# Use a preset configuration
python pf_main_v2.py --preset quick_test    # 3 agents, 5 rounds
python pf_main_v2.py --preset standard      # 12 agents, 20 rounds  
python pf_main_v2.py --preset extended      # 15 agents, 30 rounds
python pf_main_v2.py --preset high_pressure # Intense crisis period

# Override specific settings
python pf_main_v2.py --agents 15 --rounds 25

# Validate configuration without running
python pf_main_v2.py --validate-only
```

### Configuration Options

Key settings in `config.py`:

```python
# API Settings
API_PROVIDER = 'openai'              # 'openai', 'azure', 'openrouter', 'kluster'
MODEL_NAMES = {
    'openai': 'gpt-4',               # or 'gpt-4-turbo', 'gpt-3.5-turbo'
    'azure': 'your-deployment-name',
    # ...
}

# Simulation Parameters
NUM_AGENTS = 12                      # Number of agents (10-15 recommended)
NUM_ROUNDS = 20                      # Total simulation rounds
CRISIS_START_ROUND = 15              # When crisis begins
CRISIS_END_ROUND = 17                # When crisis ends

# Game Mechanics  
BASE_TOKENS_PER_ROUND = 10           # Normal token allocation
LOW_REP_TOKENS = 7                   # Tokens for low reputation
NORMAL_MULTIPLIER = 1.5              # Public pool multiplier
CRISIS_MULTIPLIER = 2.0              # Crisis multiplier
REPUTATION_THRESHOLD = 5             # Low reputation cutoff

# Family Needs
NORMAL_FAMILY_NEEDS = 4              # Tokens for "well-fed"
CRISIS_FAMILY_NEEDS = 6              # Tokens needed during crisis

# Communication
ENABLE_PUBLIC_STATEMENTS = True      # Public statements each round
ENABLE_GROUP_CHAT = True            # Group discussions
GROUP_CHAT_ROUNDS = [5, 10, 14, 18] # When chats occur

# Output
SAVE_RESULTS = True                  # Save to JSON
VERBOSE = True                       # Detailed console output
LOG_PRIVATE_INFO = True             # Include private data in logs
```

## File Structure

```
code/
├── config.py           # Central configuration file (edit this!)
├── setup_config.py     # Interactive configuration setup
├── pf_main_v2.py       # Main entry point (uses config.py)
├── pf_parameters.py    # Game parameters (set by config.py)
├── pf_agent.py         # Agent class with decision logic
├── pf_environment.py   # Game environment and coordination
├── pf_analyzer.py      # Post-simulation analysis tool
└── pf_results/         # Simulation results (auto-created)

# Legacy files (kept for compatibility)
├── pf_main.py          # Original CLI-based main file
```

## Understanding Results

### During Simulation
The simulation provides real-time feedback:
- Token allocation based on reputation
- Individual contributions and statements
- Family status (if private logging enabled)
- Reputation updates each round

### Saved Results
Results are saved as JSON with:
- Complete round-by-round data
- Agent summaries (public and private)
- Preference falsification analysis
- Behavioral pattern identification

### Analysis Tool

```bash
# Analyze results
python pf_analyzer.py pf_results/pf_simulation_20240115_143022.json

# Export analysis
python pf_analyzer.py results.json --export analysis.json
```

The analyzer identifies:
- **Behavioral Archetypes**: reputation_maximizer, family_prioritizer, etc.
- **Critical Moments**: Crisis onset, mass defections, extreme sacrifices
- **Patterns**: Contribution trends, reputation distribution, statement sentiment

## Behavioral Patterns

### Common Agent Types
1. **Reputation Maximizers**: Sacrifice family welfare for high reputation
2. **Family Prioritizers**: Keep tokens despite reputation cost
3. **Early Cooperator-Defectors**: Start generous, become selfish
4. **Adaptive Strategists**: Change strategy based on conditions

### Preference Falsification Indicators
- High reputation with suffering family
- Positive public statements while reducing contributions
- Crisis-induced behavioral changes
- Gap between public support and private actions

## Customization

### Modify Parameters
Edit `pf_parameters.py` to adjust:
- Token amounts and multipliers
- Family need levels
- Reputation thresholds
- Crisis timing and intensity
- Communication settings

### Extend Agent Behavior
The `PFAgent` class can be extended with:
- Different decision strategies
- Memory of other agents' behavior
- Coalition formation
- Punishment mechanisms

## Research Applications

This simulation is useful for studying:
- Preference falsification in social dilemmas
- Reputation systems and their effects
- Crisis response in communities
- Public vs. private incentive alignment
- LLM behavior in complex social scenarios

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variable is set correctly
2. **Rate Limits**: Reduce num-agents or add delays
3. **Memory Issues**: Lower num-rounds or disable logging
4. **JSON Parsing**: Check LLM responses in debug mode

### Debug Mode
Add verbose output:
```python
# In pf_agent.py, after getting LLM response:
print(f"DEBUG {self.name}: {response}")
```

## Citation

If using this simulation for research, please cite:
```
Preference Falsification Simulation
A framework for studying public-private incentive conflicts in LLM agents
[Your repository/paper details]
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional agent strategies
- Visualization tools
- Network effects between agents
- Alternative game mechanics
- Performance optimizations
