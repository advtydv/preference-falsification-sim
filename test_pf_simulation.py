# test_pf_simulation.py
"""
Test script for the Preference Falsification simulation.
Runs a minimal simulation to verify all components work correctly.
"""

import sys
import os
import json
from unittest.mock import Mock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pf_parameters as params
from pf_agent import PFAgent
from pf_environment import PFEnvironment

class MockAPIClient:
    """Mock API client for testing without real API calls."""
    
    def __init__(self):
        self.deployment_name = 'mock-model'
        self.model_name = 'mock-model'
        self.call_count = 0
        self.total_cost = 0.0
    
    def send_request(self, model_name, prompt):
        """Return mock responses based on prompt content."""
        self.call_count += 1
        self.total_cost += 0.01  # Mock cost
        
        # Determine response based on prompt content
        if "decide how many tokens to contribute" in prompt.lower():
            # Contribution decision
            contribution = 5 if self.call_count % 3 == 0 else 6
            statement = "I believe in supporting our community" if self.call_count % 4 == 0 else None
            
            response = {
                "contribution": contribution,
                "public_statement": statement,
                "private_reasoning": "I need to balance family and reputation",
                "public_reasoning": "Contributing to the common good"
            }
        
        elif "group discussion" in prompt.lower():
            # Group chat
            response = {
                "message": "We should all contribute fairly" if self.call_count % 2 == 0 else None
            }
        
        else:
            # Default response
            response = {"message": "Default response"}
        
        return json.dumps(response)
    
    def get_total_cost(self):
        """Return mock total cost."""
        return self.total_cost

def test_agent_creation():
    """Test agent creation and basic functionality."""
    print("Testing agent creation...")
    
    client = MockAPIClient()
    agent = PFAgent(agent_id=0, api_client=client, name="TestAgent")
    
    assert agent.agent_id == 0
    assert agent.name == "TestAgent"
    assert agent.reputation == params.INITIAL_REPUTATION
    assert agent.family_status == 'well_fed'
    assert agent.current_tokens == params.BASE_TOKENS_PER_ROUND
    
    print("✓ Agent creation successful")

def test_contribution_decision():
    """Test agent contribution decision process."""
    print("\nTesting contribution decision...")
    
    client = MockAPIClient()
    agent = PFAgent(agent_id=0, api_client=client, name="TestAgent")
    
    # Mock game state
    game_state = {
        'round': 1,
        'is_crisis': False,
        'multiplier': params.NORMAL_MULTIPLIER,
        'num_agents': 3,
        'agents_public_data': [
            {'agent_id': 0, 'name': 'TestAgent', 'reputation': 5.0, 'recent_contributions': []},
            {'agent_id': 1, 'name': 'Agent1', 'reputation': 6.0, 'recent_contributions': [6, 5, 7]},
            {'agent_id': 2, 'name': 'Agent2', 'reputation': 4.0, 'recent_contributions': [3, 4, 4]}
        ],
        'recent_statements': []
    }
    
    contribution, statement = agent.decide_contribution(game_state)
    
    assert 0 <= contribution <= agent.current_tokens
    assert contribution in [5, 6]  # Based on mock responses
    
    print(f"✓ Agent decided to contribute {contribution} tokens")
    if statement:
        print(f"  Statement: '{statement}'")

def test_reputation_update():
    """Test reputation calculation."""
    print("\nTesting reputation update...")
    
    client = MockAPIClient()
    agent = PFAgent(agent_id=0, api_client=client, name="TestAgent")
    
    # Simulate contribution history
    contributions = [6, 5, 7, 4, 8]
    for c in contributions:
        agent.contribution_history.append(c)
    
    agent.update_reputation()
    expected_reputation = sum(contributions) / len(contributions)
    
    assert agent.reputation == expected_reputation
    print(f"✓ Reputation correctly calculated: {agent.reputation:.2f}")

def test_family_status():
    """Test family status calculation."""
    print("\nTesting family status...")
    
    client = MockAPIClient()
    agent = PFAgent(agent_id=0, api_client=client, name="TestAgent")
    
    # Test different token keeping scenarios
    test_cases = [
        (4, False, 'well_fed'),     # Keep 4 tokens, normal period
        (3, False, 'surviving'),    # Keep 3 tokens
        (2, False, 'suffering'),    # Keep 2 tokens
        (1, False, 'starving'),     # Keep 1 token
        (6, True, 'well_fed'),      # Keep 6 tokens, crisis period
        (4, True, 'surviving'),     # Keep 4 tokens during crisis (4 >= 3)
    ]
    
    for tokens_kept, is_crisis, expected_status in test_cases:
        agent.is_crisis = is_crisis
        status = agent._calculate_family_status(tokens_kept)
        assert status == expected_status
        print(f"✓ Tokens kept: {tokens_kept}, Crisis: {is_crisis} → Status: {status}")

def test_mini_simulation():
    """Test a complete mini simulation."""
    print("\nTesting mini simulation (3 agents, 5 rounds)...")
    
    # Override parameters for quick test
    original_num_agents = params.NUM_AGENTS
    original_num_rounds = params.NUM_ROUNDS
    original_save = params.SAVE_RESULTS
    original_verbose = params.VERBOSE
    
    params.NUM_AGENTS = 3
    params.NUM_ROUNDS = 5
    params.SAVE_RESULTS = False
    params.VERBOSE = False
    params.CRISIS_START_ROUND = 4
    params.CRISIS_END_ROUND = 5
    params.GROUP_CHAT_ROUNDS = [3]
    
    try:
        # Create agents
        client = MockAPIClient()
        agents = [
            PFAgent(agent_id=i, api_client=client, name=f"Agent{i}")
            for i in range(params.NUM_AGENTS)
        ]
        
        # Create and run environment
        env = PFEnvironment(agents)
        env.run_simulation()
        
        # Verify results
        assert len(env.round_history) == params.NUM_ROUNDS
        assert env.round_history[3]['is_crisis'] == True  # Round 4 is crisis
        assert all(agent.reputation >= 0 for agent in agents)
        assert client.call_count > 0  # API was called
        
        print(f"✓ Simulation completed successfully")
        print(f"  Total API calls: {client.call_count}")
        print(f"  Mock cost: ${client.get_total_cost():.2f}")
        
        # Check for preference falsification patterns
        pf_analysis = env.results.get('preference_falsification_analysis', {})
        if pf_analysis:
            print(f"  Sacrifice events: {len(pf_analysis.get('sacrifice_events', []))}")
            print(f"  Defection events: {len(pf_analysis.get('defection_events', []))}")
        
    finally:
        # Restore original parameters
        params.NUM_AGENTS = original_num_agents
        params.NUM_ROUNDS = original_num_rounds
        params.SAVE_RESULTS = original_save
        params.VERBOSE = original_verbose

def main():
    """Run all tests."""
    print("="*60)
    print("PREFERENCE FALSIFICATION SIMULATION TEST SUITE")
    print("="*60)
    
    tests = [
        test_agent_creation,
        test_contribution_decision,
        test_reputation_update,
        test_family_status,
        test_mini_simulation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ All tests passed! The simulation is ready to run.")
        print("\nTo run a real simulation:")
        print("  python pf_main.py --api-provider openai --model-name gpt-4")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())