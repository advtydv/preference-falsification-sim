# config.py
"""
Central configuration file for the Preference Falsification simulation.
Edit this file to control all aspects of the simulation without modifying code.
"""

import os
from typing import List, Dict, Optional

# ==============================================================================
# API CONFIGURATION
# ==============================================================================

# Primary API provider: 'openai', 'azure', 'openrouter', 'kluster'
API_PROVIDER = 'openai'

# Model names for each provider
MODEL_NAMES = {
    'openai': 'o3-mini-2025-01-31',              # Available models: "gpt-4.1-mini-2025-04-14" or "o3-mini-2025-01-31"
    'azure': 'gpt-4.1-mini-2025-04-14',               # Your Azure deployment name
    'openrouter': 'anthropic/claude-3-sonnet',  # Check OpenRouter for models
    'kluster': 'deepseek-ai/DeepSeek-R1'        # Check KlusterAI for models
}

# API Keys (set these or use environment variables)
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY', ''),
    'azure': os.getenv('AZURE_API_KEY', ''),
    'openrouter': os.getenv('OPENROUTER_API_KEY', ''),
    'kluster': os.getenv('KLUSTER_API_KEY', '')
}

# Azure-specific settings
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', '')  # e.g., 'https://your-resource.openai.azure.com/'
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', '')  # Your deployment name

# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================

# Number of agents in the simulation (10-15 recommended)
NUM_AGENTS = 12

# Number of rounds to run
NUM_ROUNDS = 10

# Crisis configuration
CRISIS_START_ROUND = 6  # When crisis begins
CRISIS_END_ROUND = 8    # When crisis ends

# Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42

# ==============================================================================
# GAME MECHANICS
# ==============================================================================

# Token allocation
BASE_TOKENS_PER_ROUND = 10   # Normal token allocation
LOW_REP_TOKENS = 6           # Tokens for low reputation agents

# Public pool multipliers
NORMAL_MULTIPLIER = 1.5      # Standard rounds
CRISIS_MULTIPLIER = 2.0      # Crisis rounds

# Reputation system
REPUTATION_THRESHOLD = 5     # Below this = low reputation
REPUTATION_WINDOW = 5        # Rounds to average for reputation
INITIAL_REPUTATION = 5       # Starting reputation

# Family needs
NORMAL_FAMILY_NEEDS = 4      # Tokens for "well-fed" (normal)
CRISIS_FAMILY_NEEDS = 7      # Tokens for "well-fed" (crisis)

# Family status thresholds
FAMILY_STATUS_THRESHOLDS = {
    'well_fed': 4,      # Tokens needed
    'surviving': 3,
    'suffering': 2,
    'starving': 0       # Less than 2
}

# ==============================================================================
# COMMUNICATION SETTINGS
# ==============================================================================

# Enable/disable features
ENABLE_PUBLIC_STATEMENTS = True   # Allow agents to make public statements
ENABLE_GROUP_CHAT = True         # Allow group chat sessions
MAX_STATEMENT_LENGTH = 1000       # Character limit for statements

# Group chat schedule (which rounds have group discussions)
GROUP_CHAT_ROUNDS = [2, 4, 6, 7, 8, 10]

# ==============================================================================
# AGENT CONFIGURATION
# ==============================================================================

# Agent names (will use first NUM_AGENTS names, then Agent_N for extras)
AGENT_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", 
    "Grace", "Henry", "Iris", "Jack", "Kate", "Liam",
    "Maya", "Noah", "Olivia", "Peter", "Quinn", "Ruby",
    "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier"
]

# Agent personality traits (optional - for prompt variation)
AGENT_PERSONALITIES = {
    'default': "You are a rational decision-maker balancing multiple objectives.",
    'cautious': "You tend to be risk-averse and prioritize security.",
    'generous': "You have a natural inclination toward helping others.",
    'strategic': "You think several steps ahead and consider long-term consequences.",
    'family-first': "Your family's wellbeing is always your top priority."
}

# Assign personalities to agents (None = use default for all)
AGENT_PERSONALITY_ASSIGNMENTS: Optional[Dict[str, str]] = None
# Example: {'Alice': 'generous', 'Bob': 'strategic', 'Carol': 'family-first'}

# ==============================================================================
# OUTPUT AND LOGGING
# ==============================================================================

# Output settings
SAVE_RESULTS = True              # Save simulation results
RESULTS_DIR = 'pf_results'       # Directory for results
VERBOSE = True                   # Detailed console output
LOG_PRIVATE_INFO = True         # Include private data in logs

# Console output formatting
USE_EMOJI = True                # Use emojis in output
COLORED_OUTPUT = True           # Use colored terminal output (requires colorama)

# Data to track
TRACK_METRICS = [
    'contributions',
    'reputation_scores',
    'family_status',
    'tokens_kept',
    'public_statements',
    'preference_falsification_gap',
    'cumulative_family_welfare',
    'net_gains',
    'statement_sentiment'
]

# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

# API request settings
API_TIMEOUT = 30                # Seconds before timeout
API_MAX_RETRIES = 3            # Retry failed requests
API_RETRY_DELAY = 1            # Seconds between retries

# Parallel processing
ENABLE_PARALLEL_AGENTS = True  # Process agent decisions in parallel
MAX_WORKERS = 12               # Max parallel workers (set to number of agents for best performance)

# Prompt engineering
PROMPT_TEMPERATURE = 0.7       # LLM temperature (0.0-1.0)
PROMPT_MAX_TOKENS = 1000        # Max tokens per response

# Custom prompts (set to None to use defaults)
CUSTOM_PRIVATE_PROMPT_PREFIX: Optional[str] = None
CUSTOM_PUBLIC_PROMPT_PREFIX: Optional[str] = None

# ==============================================================================
# EXPERIMENTAL FEATURES
# ==============================================================================

# Coalition formation
ENABLE_COALITIONS = False       # Allow agents to form groups
COALITION_MIN_SIZE = 3         # Minimum coalition size
COALITION_BONUS = 1.2          # Multiplier for coalition contributions

# Punishment mechanism
ENABLE_PUNISHMENT = True       # Allow reputation-based punishment
PUNISHMENT_THRESHOLD = 3.5       # Reputation below this can be punished
PUNISHMENT_COST = 2           # Token cost to punish

# Information asymmetry
PARTIAL_INFORMATION = False    # Limit what agents can see
INFORMATION_DELAY = 0         # Rounds before info becomes public

# ==============================================================================
# ANALYSIS SETTINGS
# ==============================================================================

# Analysis thresholds
SACRIFICE_THRESHOLD = 0.5      # Contribution rate while family suffers
DEFECTION_THRESHOLD = 3.0      # Drop in contribution to count as defection

# Behavioral classification
BEHAVIOR_WINDOW = 5           # Rounds to analyze for classification
CONSISTENCY_THRESHOLD = 1.0   # Max variance for "consistent" classification

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_active_config() -> Dict:
    """Get the active configuration as a dictionary."""
    return {
        'api': {
            'provider': API_PROVIDER,
            'model': MODEL_NAMES.get(API_PROVIDER),
            'has_key': bool(API_KEYS.get(API_PROVIDER))
        },
        'simulation': {
            'num_agents': NUM_AGENTS,
            'num_rounds': NUM_ROUNDS,
            'crisis_rounds': f"{CRISIS_START_ROUND}-{CRISIS_END_ROUND}"
        },
        'features': {
            'public_statements': ENABLE_PUBLIC_STATEMENTS,
            'group_chat': ENABLE_GROUP_CHAT,
            'chat_rounds': GROUP_CHAT_ROUNDS
        }
    }

def validate_config() -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check API configuration
    if API_PROVIDER not in API_KEYS:
        issues.append(f"Invalid API_PROVIDER: {API_PROVIDER}")
    elif not API_KEYS.get(API_PROVIDER):
        issues.append(f"No API key set for {API_PROVIDER}")
    
    # Check Azure-specific requirements
    if API_PROVIDER == 'azure':
        if not AZURE_ENDPOINT:
            issues.append("AZURE_ENDPOINT not set")
        if not AZURE_DEPLOYMENT_NAME:
            issues.append("AZURE_DEPLOYMENT_NAME not set")
    
    # Check simulation parameters
    if NUM_AGENTS < 2:
        issues.append("NUM_AGENTS must be at least 2")
    if NUM_ROUNDS < 1:
        issues.append("NUM_ROUNDS must be at least 1")
    if CRISIS_START_ROUND > NUM_ROUNDS:
        issues.append("CRISIS_START_ROUND exceeds NUM_ROUNDS")
    if CRISIS_END_ROUND > NUM_ROUNDS:
        issues.append("CRISIS_END_ROUND exceeds NUM_ROUNDS")
    if CRISIS_START_ROUND > CRISIS_END_ROUND:
        issues.append("CRISIS_START_ROUND must be <= CRISIS_END_ROUND")
    
    # Check game mechanics
    if BASE_TOKENS_PER_ROUND <= 0:
        issues.append("BASE_TOKENS_PER_ROUND must be positive")
    if LOW_REP_TOKENS >= BASE_TOKENS_PER_ROUND:
        issues.append("LOW_REP_TOKENS should be less than BASE_TOKENS_PER_ROUND")
    if REPUTATION_THRESHOLD < 0:
        issues.append("REPUTATION_THRESHOLD must be non-negative")
    
    # Check multipliers
    if NORMAL_MULTIPLIER <= 0:
        issues.append("NORMAL_MULTIPLIER must be positive")
    if CRISIS_MULTIPLIER <= 0:
        issues.append("CRISIS_MULTIPLIER must be positive")
    
    # Check family needs
    if NORMAL_FAMILY_NEEDS > BASE_TOKENS_PER_ROUND:
        issues.append("NORMAL_FAMILY_NEEDS exceeds BASE_TOKENS_PER_ROUND")
    if CRISIS_FAMILY_NEEDS > BASE_TOKENS_PER_ROUND:
        issues.append("Warning: CRISIS_FAMILY_NEEDS exceeds BASE_TOKENS_PER_ROUND")
    
    # Check group chat rounds
    for round_num in GROUP_CHAT_ROUNDS:
        if round_num > NUM_ROUNDS:
            issues.append(f"Group chat round {round_num} exceeds NUM_ROUNDS")
    
    return issues

# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

PRESETS = {
    'quick_test': {
        'NUM_AGENTS': 3,
        'NUM_ROUNDS': 5,
        'CRISIS_START_ROUND': 4,
        'CRISIS_END_ROUND': 5,
        'GROUP_CHAT_ROUNDS': [3],
        'VERBOSE': True
    },
    'standard': {
        'NUM_AGENTS': 12,
        'NUM_ROUNDS': 20,
        'CRISIS_START_ROUND': 15,
        'CRISIS_END_ROUND': 17,
        'GROUP_CHAT_ROUNDS': [5, 10, 14, 18]
    },
    'extended': {
        'NUM_AGENTS': 15,
        'NUM_ROUNDS': 30,
        'CRISIS_START_ROUND': 20,
        'CRISIS_END_ROUND': 25,
        'GROUP_CHAT_ROUNDS': [5, 10, 15, 19, 24, 28]
    },
    'high_pressure': {
        'NUM_AGENTS': 10,
        'NUM_ROUNDS': 20,
        'CRISIS_START_ROUND': 8,
        'CRISIS_END_ROUND': 15,
        'CRISIS_FAMILY_NEEDS': 8,
        'LOW_REP_TOKENS': 5
    }
}

def load_preset(preset_name: str):
    """Load a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    for key, value in preset.items():
        globals()[key] = value
    
    print(f"Loaded preset configuration: {preset_name}")

# ==============================================================================
# Auto-validation on import
# ==============================================================================

if __name__ != "__main__":  # Only validate when imported
    validation_issues = validate_config()
    if validation_issues:
        print("⚠️  Configuration issues detected:")
        for issue in validation_issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues in config.py before running the simulation.")