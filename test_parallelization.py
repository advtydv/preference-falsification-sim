#!/usr/bin/env python3
"""
Test script to verify if parallelization is working correctly.
"""

import time
import pf_parameters as params
import config
from pf_environment import PFEnvironment
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_parallel_timing():
    """Test if parallel execution is actually faster than sequential."""
    print("🔍 Testing Parallelization Performance...")
    print("="*50)
    
    # Store original settings
    original_parallel = params.ENABLE_PARALLEL_AGENTS
    original_workers = params.MAX_WORKERS
    original_agents = params.NUM_AGENTS
    
    # Test with a reasonable number of agents
    params.NUM_AGENTS = 8
    params.NUM_ROUNDS = 1
    params.VERBOSE = False
    
    # Import necessary modules
    from pf_agent import PFAgent
    from openai_client import OpenAIClient
    
    # Create API client
    api_key = config.API_KEYS.get('openai')
    if not api_key:
        print("❌ No OpenAI API key found in config.py")
        return
    
    api_client = OpenAIClient(api_key=api_key, model_name=config.MODEL_NAMES['openai'], reasoning_effort='low')
    
    try:
        # Create agents
        agents = []
        for i in range(params.NUM_AGENTS):
            agent = PFAgent(agent_id=i, api_client=api_client, name=f"Agent_{i}")
            agents.append(agent)
        
        # Test sequential execution
        print("\n⏱️  Testing Sequential Execution...")
        params.ENABLE_PARALLEL_AGENTS = False
        env_seq = PFEnvironment(agents)
        
        start_time = time.time()
        env_seq.run_round()
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f} seconds")
        
        # Create new agents for parallel test (to avoid state issues)
        agents_par = []
        for i in range(params.NUM_AGENTS):
            agent = PFAgent(agent_id=i, api_client=api_client, name=f"Agent_{i}")
            agents_par.append(agent)
        
        # Test parallel execution
        print("\n⏱️  Testing Parallel Execution...")
        params.ENABLE_PARALLEL_AGENTS = True
        params.MAX_WORKERS = params.NUM_AGENTS  # One worker per agent
        env_par = PFEnvironment(agents_par)
        
        start_time = time.time()
        env_par.run_round()
        parallel_time = time.time() - start_time
        print(f"Parallel time: {parallel_time:.2f} seconds")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\n📊 Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ Parallelization is working effectively!")
        elif speedup > 1.1:
            print("⚠️  Parallelization provides modest improvement")
        else:
            print("❌ Parallelization is NOT providing speedup")
        
    finally:
        # Restore original settings
        params.ENABLE_PARALLEL_AGENTS = original_parallel
        params.MAX_WORKERS = original_workers
        params.NUM_AGENTS = original_agents
        params.VERBOSE = True

def test_thread_pool_usage():
    """Verify that ThreadPoolExecutor is actually using multiple threads."""
    print("\n\n🔍 Testing Thread Pool Usage...")
    print("="*50)
    
    import threading
    active_threads = set()
    
    def track_thread(name):
        """Function to track which threads are being used."""
        thread_id = threading.current_thread().ident
        active_threads.add(thread_id)
        time.sleep(0.1)  # Simulate work
        return f"{name} executed on thread {thread_id}"
    
    # Test with multiple workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(track_thread, f"Task_{i}") for i in range(8)]
        
        for future in as_completed(futures):
            result = future.result()
            print(f"  {result}")
    
    print(f"\n📊 Number of unique threads used: {len(active_threads)}")
    if len(active_threads) > 1:
        print("✅ ThreadPoolExecutor is using multiple threads")
    else:
        print("❌ ThreadPoolExecutor is NOT using multiple threads")

def test_actual_llm_parallelization():
    """Test if LLM calls are truly being made in parallel."""
    print("\n\n🔍 Testing Actual LLM Call Parallelization...")
    print("="*50)
    
    # Create a simple test that tracks API call timing
    from pf_agent import PFAgent
    from openai_client import OpenAIClient
    import threading
    
    call_times = []
    call_lock = threading.Lock()
    
    # Create API client
    api_key = config.API_KEYS.get('openai')
    if not api_key:
        print("❌ No OpenAI API key found in config.py")
        return
    
    api_client = OpenAIClient(api_key=api_key, model_name=config.MODEL_NAMES['openai'], reasoning_effort='low')
    
    # Monkey patch the API client to track call times
    original_send_request = api_client.send_request
    
    def tracked_send_request(*args, **kwargs):
        thread_id = threading.current_thread().ident
        start_time = time.time()
        
        with call_lock:
            call_idx = len(call_times)
            call_times.append({
                'thread': thread_id,
                'start': start_time,
                'end': None
            })
        
        result = original_send_request(*args, **kwargs)
        
        end_time = time.time()
        with call_lock:
            call_times[call_idx]['end'] = end_time
        
        return result
    
    # Patch the send_request method
    api_client.send_request = tracked_send_request
    
    # Run a quick test
    params.NUM_AGENTS = 4
    params.ENABLE_PARALLEL_AGENTS = True
    params.MAX_WORKERS = 4
    params.VERBOSE = False
    
    # Create agents
    agents = []
    for i in range(params.NUM_AGENTS):
        agent = PFAgent(agent_id=i, api_client=api_client, name=f"Agent_{i}")
        agents.append(agent)
    
    env = PFEnvironment(agents)
    
    # Run one round
    public_game_state = env._construct_public_game_state()
    
    with ThreadPoolExecutor(max_workers=params.MAX_WORKERS) as executor:
        futures = {
            executor.submit(env._process_agent_decision, agent, public_game_state): agent 
            for agent in env.agents
        }
        
        for future in as_completed(futures):
            try:
                future.result()
            except:
                pass
    
    # Analyze timing
    if call_times:
        # Check for overlapping API calls
        overlapping_calls = 0
        for i in range(len(call_times)):
            for j in range(i + 1, len(call_times)):
                # Check if calls overlap in time
                if (call_times[i]['start'] < call_times[j]['end'] and 
                    call_times[j]['start'] < call_times[i]['end']):
                    overlapping_calls += 1
        
        unique_threads = len(set(ct['thread'] for ct in call_times))
        
        print(f"\n📊 Analysis:")
        print(f"  Total API calls: {len(call_times)}")
        print(f"  Unique threads: {unique_threads}")
        print(f"  Overlapping calls: {overlapping_calls}")
        
        if overlapping_calls > 0:
            print("✅ LLM calls are being made in parallel!")
        else:
            print("❌ LLM calls are sequential (no time overlap detected)")
    else:
        print("❌ No API calls were tracked")

if __name__ == "__main__":
    print("🚀 Parallelization Test Suite")
    print("="*70)
    
    test_thread_pool_usage()
    test_parallel_timing()
    test_actual_llm_parallelization()
    
    print("\n\n📌 Summary:")
    print("If parallelization is NOT working, check:")
    print("  1. config.py: ENABLE_PARALLEL_AGENTS should be True")
    print("  2. config.py: MAX_WORKERS should be >= NUM_AGENTS")
    print("  3. Python GIL limitations (CPU-bound vs I/O-bound)")
    print("  4. API rate limits or throttling")