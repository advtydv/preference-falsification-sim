# pf_main_v2.py
"""
Main entry point for the Preference Falsification simulation.
Uses centralized configuration from config.py.
"""

import os
import sys
import argparse
import random
from typing import List

# Import configuration
import config

# Validate config on import
validation_issues = config.validate_config()
if validation_issues:
    print("❌ Configuration errors:")
    for issue in validation_issues:
        print(f"   - {issue}")
    sys.exit(1)

# Import simulation components
from pf_agent import PFAgent
from pf_environment import PFEnvironment

# Import API clients
from openai_client import OpenAIClient
from azure_openai_client import AzureOpenAIClient
from openrouter_client import OpenRouterClient

# Try to import KlusterAI client (may not exist)
try:
    from kluster_ai_client import KlusterAIClient
except ImportError:
    KlusterAIClient = None

def initialize_api_client():
    """Initialize the API client based on config settings."""
    provider = config.API_PROVIDER
    api_key = config.API_KEYS.get(provider)
    model_name = config.MODEL_NAMES.get(provider)
    
    if not api_key:
        raise ValueError(f"No API key configured for {provider}. Set it in config.py or environment.")
    
    print(f"🔧 Initializing {provider} API client...")
    print(f"   Model: {model_name}")
    
    if provider == 'openai':
        return OpenAIClient(
            api_key=api_key, 
            model_name=model_name,
            reasoning_effort='low'
        )
    
    elif provider == 'azure':
        if not config.AZURE_ENDPOINT or not config.AZURE_DEPLOYMENT_NAME:
            raise ValueError("Azure requires AZURE_ENDPOINT and AZURE_DEPLOYMENT_NAME in config.py")
        return AzureOpenAIClient(
            api_key=api_key,
            endpoint=config.AZURE_ENDPOINT,
            deployment_name=config.AZURE_DEPLOYMENT_NAME
        )
    
    elif provider == 'openrouter':
        return OpenRouterClient(
            api_key=api_key,
            model_name=model_name
        )
    
    elif provider == 'kluster':
        if KlusterAIClient is None:
            raise ValueError("KlusterAI client not available. Please use another provider.")
        return KlusterAIClient(
            api_key=api_key,
            model_name=model_name
        )
    
    else:
        raise ValueError(f"Unsupported API provider: {provider}")

def create_agents(api_client) -> List[PFAgent]:
    """Create agents based on configuration."""
    agents = []
    
    # Set random seed if specified
    if config.RANDOM_SEED is not None:
        random.seed(config.RANDOM_SEED)
    
    print(f"\n🤖 Creating {config.NUM_AGENTS} agents...")
    
    for i in range(config.NUM_AGENTS):
        # Get agent name
        if i < len(config.AGENT_NAMES):
            name = config.AGENT_NAMES[i]
        else:
            name = f"Agent_{i}"
        
        # Create agent
        agent = PFAgent(agent_id=i, api_client=api_client, name=name)
        
        # Set personality if configured
        if config.AGENT_PERSONALITY_ASSIGNMENTS:
            personality = config.AGENT_PERSONALITY_ASSIGNMENTS.get(name)
            if personality and personality in config.AGENT_PERSONALITIES:
                agent.personality = config.AGENT_PERSONALITIES[personality]
        
        agents.append(agent)
        print(f"   ✓ Created {name}")
    
    return agents

def update_parameters():
    """Update pf_parameters module with config values."""
    import pf_parameters as params
    
    # Update all parameters from config
    params.NUM_AGENTS = config.NUM_AGENTS
    params.NUM_ROUNDS = config.NUM_ROUNDS
    params.CRISIS_START_ROUND = config.CRISIS_START_ROUND
    params.CRISIS_END_ROUND = config.CRISIS_END_ROUND
    params.SEED = config.RANDOM_SEED
    
    params.BASE_TOKENS_PER_ROUND = config.BASE_TOKENS_PER_ROUND
    params.LOW_REP_TOKENS = config.LOW_REP_TOKENS
    params.NORMAL_MULTIPLIER = config.NORMAL_MULTIPLIER
    params.CRISIS_MULTIPLIER = config.CRISIS_MULTIPLIER
    
    params.REPUTATION_THRESHOLD = config.REPUTATION_THRESHOLD
    params.REPUTATION_WINDOW = config.REPUTATION_WINDOW
    params.INITIAL_REPUTATION = config.INITIAL_REPUTATION
    
    params.NORMAL_FAMILY_NEEDS = config.NORMAL_FAMILY_NEEDS
    params.CRISIS_FAMILY_NEEDS = config.CRISIS_FAMILY_NEEDS
    
    params.ENABLE_PUBLIC_STATEMENTS = config.ENABLE_PUBLIC_STATEMENTS
    params.ENABLE_GROUP_CHAT = config.ENABLE_GROUP_CHAT
    params.MAX_STATEMENT_LENGTH = config.MAX_STATEMENT_LENGTH
    params.GROUP_CHAT_ROUNDS = config.GROUP_CHAT_ROUNDS
    
    params.SAVE_RESULTS = config.SAVE_RESULTS
    params.RESULTS_DIR = config.RESULTS_DIR
    params.VERBOSE = config.VERBOSE
    params.LOG_PRIVATE_INFO = config.LOG_PRIVATE_INFO
    
    params.TRACK_METRICS = config.TRACK_METRICS

def print_config_summary():
    """Print a summary of the active configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    active = config.get_active_config()
    
    print(f"\n🔌 API Configuration:")
    print(f"   Provider: {active['api']['provider']}")
    print(f"   Model: {active['api']['model']}")
    print(f"   API Key: {'✓ Set' if active['api']['has_key'] else '✗ Missing'}")
    
    print(f"\n🎮 Simulation Settings:")
    print(f"   Agents: {active['simulation']['num_agents']}")
    print(f"   Rounds: {active['simulation']['num_rounds']}")
    print(f"   Crisis: Rounds {active['simulation']['crisis_rounds']}")
    
    print(f"\n💬 Communication Features:")
    print(f"   Public Statements: {'✓ Enabled' if active['features']['public_statements'] else '✗ Disabled'}")
    print(f"   Group Chat: {'✓ Enabled' if active['features']['group_chat'] else '✗ Disabled'}")
    if active['features']['group_chat']:
        print(f"   Chat Rounds: {active['features']['chat_rounds']}")
    
    print(f"\n📊 Game Mechanics:")
    print(f"   Normal: {config.BASE_TOKENS_PER_ROUND} tokens/round, {config.NORMAL_MULTIPLIER}x multiplier")
    print(f"   Crisis: {config.CRISIS_MULTIPLIER}x multiplier, {config.CRISIS_FAMILY_NEEDS} tokens needed")
    print(f"   Low Rep: < {config.REPUTATION_THRESHOLD} → {config.LOW_REP_TOKENS} tokens")
    
    print("="*60 + "\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Preference Falsification simulation with config.py settings.",
        epilog="Edit config.py to change simulation parameters."
    )
    
    # Preset loading
    parser.add_argument(
        '--preset', 
        choices=list(config.PRESETS.keys()),
        help='Load a preset configuration'
    )
    
    # Override options (these override config.py)
    parser.add_argument('--agents', type=int, help='Override NUM_AGENTS')
    parser.add_argument('--rounds', type=int, help='Override NUM_ROUNDS')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--validate-only', action='store_true', help='Validate config and exit')
    
    args = parser.parse_args()
    
    # Load preset if specified
    if args.preset:
        config.load_preset(args.preset)
        # Re-validate after loading preset
        validation_issues = config.validate_config()
        if validation_issues:
            print("❌ Configuration errors after loading preset:")
            for issue in validation_issues:
                print(f"   - {issue}")
            return 1
    
    # Apply command-line overrides
    if args.agents:
        config.NUM_AGENTS = args.agents
    if args.rounds:
        config.NUM_ROUNDS = args.rounds
    if args.no_save:
        config.SAVE_RESULTS = False
    if args.quiet:
        config.VERBOSE = False
    
    # Update parameters module
    update_parameters()
    
    # Validate only mode
    if args.validate_only:
        issues = config.validate_config()
        if issues:
            print("❌ Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
            return 1
        else:
            print("✅ Configuration is valid!")
            print_config_summary()
            return 0
    
    # Print configuration
    print_config_summary()
    
    try:
        # Initialize API client
        api_client = initialize_api_client()
        
        # Create agents
        agents = create_agents(api_client)
        
        # Initialize environment
        print("\n🌍 Initializing environment...")
        env = PFEnvironment(agents)
        
        # Run simulation
        env.run_simulation()
        
        # Print cost information if available
        if hasattr(api_client, 'get_total_cost'):
            total_cost = api_client.get_total_cost()
            print(f"\n💰 API Usage Summary:")
            print(f"   Total cost: ${total_cost:.4f}")
            print(f"   Per agent: ${total_cost/config.NUM_AGENTS:.4f}")
            print(f"   Per round: ${total_cost/config.NUM_ROUNDS:.4f}")
            
            # Estimate for full simulation
            if args.preset == 'quick_test':
                standard_cost = total_cost * (12/config.NUM_AGENTS) * (20/config.NUM_ROUNDS)
                print(f"\n   Estimated cost for standard simulation: ${standard_cost:.2f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if config.VERBOSE:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())