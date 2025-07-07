#!/usr/bin/env python3
"""
Test to verify contribution decisions are parallel but group chat is sequential.
"""

import time
import pf_parameters as params
import config
from pf_environment import PFEnvironment
from pf_agent import PFAgent
from openai_client import OpenAIClient

def test_mixed_execution():
    """Test that contributions are parallel but chat is sequential."""
    print("🔍 Testing Mixed Execution (Parallel Contributions, Sequential Chat)...")
    print("="*70)
    
    # Configure for testing
    params.NUM_AGENTS = 4
    params.NUM_ROUNDS = 2
    params.VERBOSE = True
    params.ENABLE_PARALLEL_AGENTS = True
    params.MAX_WORKERS = 4
    params.GROUP_CHAT_ROUNDS = [2]  # Enable chat in round 2
    
    # Create API client
    api_key = config.API_KEYS.get('openai')
    if not api_key:
        print("❌ No OpenAI API key found in config.py")
        return
    
    api_client = OpenAIClient(api_key=api_key, model_name=config.MODEL_NAMES['openai'], reasoning_effort='low')
    
    # Create agents
    agents = []
    for i in range(params.NUM_AGENTS):
        agent = PFAgent(agent_id=i, api_client=api_client, name=f"Agent_{i}")
        agents.append(agent)
    
    # Create environment
    env = PFEnvironment(agents)
    
    # Run simulation
    print("\n📋 Running 2 rounds: Round 1 (no chat) and Round 2 (with chat)")
    print("Watch for [Parallel mode] in contributions and sequential chat messages\n")
    
    env.run_simulation()
    
    print("\n✅ Test complete!")
    print("\nExpected behavior:")
    print("- Contribution decisions: [Parallel mode: X workers for Y agents]")
    print("- Group chat: Sequential messages where agents can see previous messages")

if __name__ == "__main__":
    test_mixed_execution()