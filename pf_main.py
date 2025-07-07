# pf_main.py
"""
Main entry point for the Preference Falsification simulation.
Handles command-line arguments and initializes the simulation.
"""

import os
import argparse
import random
from typing import List
import pf_parameters as params
from pf_agent import PFAgent
from pf_environment import PFEnvironment

# Import API clients
from azure_openai_client import AzureOpenAIClient
from openrouter_client import OpenRouterClient
from kluster_ai_client import KlusterAIClient
from openai_client import OpenAIClient

def create_agent_names() -> List[str]:
    """Generate diverse agent names for better readability."""
    names = [
        "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Liam", "Maya", "Noah", "Olivia", "Peter",
        "Quinn", "Ruby", "Sam", "Tara"
    ]
    return random.sample(names, min(params.NUM_AGENTS, len(names)))

def initialize_api_client(args):
    """Initialize the appropriate API client based on arguments."""
    api_provider = args.api_provider.lower()
    
    if api_provider == 'openai':
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        model_name = args.model_name or os.getenv('OPENAI_MODEL_NAME', 'gpt-4')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
        return OpenAIClient(api_key=api_key, model_name=model_name, reasoning_effort='low')
    
    elif api_provider == 'azure':
        api_key = args.api_key or os.getenv('AZURE_API_KEY')
        endpoint = args.azure_endpoint or os.getenv('AZURE_ENDPOINT')
        deployment = args.deployment_name or os.getenv('AZURE_DEPLOYMENT_NAME')
        if not all([api_key, endpoint, deployment]):
            raise ValueError("Azure credentials required. Set AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME")
        return AzureOpenAIClient(api_key=api_key, endpoint=endpoint, deployment_name=deployment)
    
    elif api_provider == 'openrouter':
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        model_name = args.model_name or 'anthropic/claude-3-sonnet'
        if not api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key")
        return OpenRouterClient(api_key=api_key, model_name=model_name)
    
    elif api_provider == 'kluster':
        api_key = args.api_key or os.getenv('KLUSTER_API_KEY')
        model_name = args.model_name or 'deepseek-ai/DeepSeek-R1'
        if not api_key:
            raise ValueError("KlusterAI API key required. Set KLUSTER_API_KEY or use --api-key")
        return KlusterAIClient(api_key=api_key, model_name=model_name)
    
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")

def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(
        description="Run the Preference Falsification simulation where agents balance public reputation with private family needs."
    )
    
    # API configuration
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument('--api-provider', type=str, default='openai',
                          choices=['openai', 'azure', 'openrouter', 'kluster'],
                          help='API provider for LLM calls (default: openai)')
    api_group.add_argument('--model-name', type=str, default=None,
                          help='Model name (provider-specific)')
    api_group.add_argument('--api-key', type=str, default=None,
                          help='API key (overrides environment variable)')
    api_group.add_argument('--azure-endpoint', type=str, default=None,
                          help='Azure OpenAI endpoint (for Azure provider)')
    api_group.add_argument('--deployment-name', type=str, default=None,
                          help='Azure deployment name (for Azure provider)')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--num-agents', type=int, default=params.NUM_AGENTS,
                          help=f'Number of agents (default: {params.NUM_AGENTS})')
    sim_group.add_argument('--num-rounds', type=int, default=params.NUM_ROUNDS,
                          help=f'Number of rounds (default: {params.NUM_ROUNDS})')
    sim_group.add_argument('--crisis-start', type=int, default=params.CRISIS_START_ROUND,
                          help=f'Crisis start round (default: {params.CRISIS_START_ROUND})')
    sim_group.add_argument('--crisis-end', type=int, default=params.CRISIS_END_ROUND,
                          help=f'Crisis end round (default: {params.CRISIS_END_ROUND})')
    
    # Feature toggles
    feature_group = parser.add_argument_group('Feature Toggles')
    feature_group.add_argument('--disable-statements', action='store_true',
                              help='Disable public statements')
    feature_group.add_argument('--disable-chat', action='store_true',
                              help='Disable group chat')
    feature_group.add_argument('--disable-private-logging', action='store_true',
                              help='Disable logging of private information')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--quiet', action='store_true',
                             help='Reduce output verbosity')
    output_group.add_argument('--results-dir', type=str, default=params.RESULTS_DIR,
                             help=f'Directory for results (default: {params.RESULTS_DIR})')
    output_group.add_argument('--no-save', action='store_true',
                             help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Update parameters based on arguments
    params.NUM_AGENTS = args.num_agents
    params.NUM_ROUNDS = args.num_rounds
    params.CRISIS_START_ROUND = args.crisis_start
    params.CRISIS_END_ROUND = args.crisis_end
    params.ENABLE_PUBLIC_STATEMENTS = not args.disable_statements
    params.ENABLE_GROUP_CHAT = not args.disable_chat
    params.LOG_PRIVATE_INFO = not args.disable_private_logging
    params.VERBOSE = not args.quiet
    params.RESULTS_DIR = args.results_dir
    params.SAVE_RESULTS = not args.no_save
    
    # Validate parameters
    if params.CRISIS_START_ROUND > params.NUM_ROUNDS:
        raise ValueError(f"Crisis start round ({params.CRISIS_START_ROUND}) cannot exceed total rounds ({params.NUM_ROUNDS})")
    if params.CRISIS_END_ROUND > params.NUM_ROUNDS:
        params.CRISIS_END_ROUND = params.NUM_ROUNDS
    
    # Initialize API client
    try:
        api_client = initialize_api_client(args)
        print(f"✓ Initialized {args.api_provider} API client")
        if hasattr(api_client, 'model_name'):
            print(f"  Model: {api_client.model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize API client: {e}")
        return 1
    
    # Create agents with diverse names
    agent_names = create_agent_names()
    agents = []
    
    print(f"\n🤖 Creating {params.NUM_AGENTS} agents...")
    for i in range(params.NUM_AGENTS):
        name = agent_names[i] if i < len(agent_names) else f"Agent_{i}"
        agent = PFAgent(agent_id=i, api_client=api_client, name=name)
        agents.append(agent)
        print(f"   ✓ Created {name}")
    
    # Initialize and run environment
    print("\n🌍 Initializing environment...")
    env = PFEnvironment(agents)
    
    try:
        env.run_simulation()
        
        # Print cost information if available
        if hasattr(api_client, 'get_total_cost'):
            total_cost = api_client.get_total_cost()
            print(f"\n💰 Total API Cost: ${total_cost:.4f}")
            print(f"   Per agent: ${total_cost/params.NUM_AGENTS:.4f}")
            print(f"   Per round: ${total_cost/params.NUM_ROUNDS:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())