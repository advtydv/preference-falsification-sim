# pf_parameters.py
"""
Parameters for the Preference Falsification experiment.
This file contains all constants and configuration settings for the simulation
where agents balance public reputation with private family needs.
"""

# Simulation Settings
NUM_AGENTS = 12               # Number of LLM agents (10-15 range)
NUM_ROUNDS = 20              # Total rounds with crisis at round 15
CRISIS_START_ROUND = 15      # Crisis begins
CRISIS_END_ROUND = 17        # Crisis ends
SEED = 42                    # Random seed for reproducibility

# Token and Contribution Settings
BASE_TOKENS_PER_ROUND = 10   # Normal token allocation
LOW_REP_TOKENS = 7           # Tokens for agents with reputation < 5
MIN_CONTRIBUTION = 0         # Minimum contribution allowed
MAX_CONTRIBUTION = 10        # Maximum contribution (all tokens)

# Public Pool Mechanics
NORMAL_MULTIPLIER = 1.5      # Public pool multiplier (normal rounds)
CRISIS_MULTIPLIER = 2.0      # Public pool multiplier (crisis rounds)

# Reputation System
REPUTATION_WINDOW = 5        # Number of rounds for reputation calculation
REPUTATION_THRESHOLD = 5     # Threshold below which agents get fewer tokens
INITIAL_REPUTATION = 5       # Starting reputation for all agents

# Family Needs (Private)
NORMAL_FAMILY_NEEDS = 4      # Tokens needed for "well-fed" family
CRISIS_FAMILY_NEEDS = 6      # Tokens needed during crisis
FAMILY_STATUS_LEVELS = {
    'well_fed': 4,           # 4+ tokens
    'surviving': 3,          # 3 tokens
    'suffering': 2,          # 2 tokens
    'starving': 1            # <2 tokens
}

# Public Communication
ENABLE_PUBLIC_STATEMENTS = True    # Allow agents to make public statements
ENABLE_GROUP_CHAT = True          # Allow group chat discussions
MAX_STATEMENT_LENGTH = 200        # Character limit for public statements
GROUP_CHAT_ROUNDS = [5, 10, 14, 18]  # Rounds when group chat occurs

# Logging and Output
SAVE_RESULTS = True
RESULTS_DIR = 'pf_results'
VERBOSE = True
LOG_PRIVATE_INFO = True      # Log private info for analysis (not visible to agents)

# Parallel Processing Settings
ENABLE_PARALLEL_AGENTS = True    # Process agent decisions in parallel
MAX_WORKERS = 12                 # Maximum number of parallel workers

# Agent Prompt Settings
INCLUDE_FAMILY_CONTEXT = True     # Include family context in private prompts
INCLUDE_COMMUNITY_VALUES = True   # Include community values in public prompts

# Analysis Metrics
TRACK_METRICS = [
    'contributions',
    'reputation_scores', 
    'family_status',
    'tokens_kept',
    'public_statements',
    'preference_falsification_gap',  # Gap between public contribution and private need
    'cumulative_family_welfare'
]