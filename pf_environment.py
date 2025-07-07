# pf_environment.py
"""
Environment class for the Preference Falsification experiment.
Manages game rounds, reputation system, crisis events, and agent coordination.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pf_parameters as params
from pf_agent import PFAgent

class PFEnvironment:
    def __init__(self, agents: List[PFAgent]):
        """
        Initialize the Preference Falsification Environment.
        
        Args:
            agents: List of PFAgent instances
        """
        self.agents = agents
        self.current_round = 0
        self.is_crisis = False
        
        # Game state tracking
        self.round_history = []
        self.public_pool = 0
        self.recent_statements = []  # Track recent public statements
        self.group_chat_history = []  # Track group chat messages
        self.terminated_agents = []  # Track agents whose families died
        self.termination_announcements = []  # Public announcements of deaths
        
        # Results storage
        self.results = {
            'parameters': self._get_parameters_dict(),
            'rounds': [],
            'agent_summaries': {},
            'preference_falsification_analysis': {}
        }
        
        # Create results directory
        if params.SAVE_RESULTS and not os.path.exists(params.RESULTS_DIR):
            os.makedirs(params.RESULTS_DIR)

    def run_simulation(self):
        """Run the complete simulation for NUM_ROUNDS."""
        print(f"\n🎮 Starting Preference Falsification Simulation")
        print(f"   Agents: {params.NUM_AGENTS}")
        print(f"   Rounds: {params.NUM_ROUNDS}")
        print(f"   Crisis: Rounds {params.CRISIS_START_ROUND}-{params.CRISIS_END_ROUND}\n")
        
        for round_num in range(1, params.NUM_ROUNDS + 1):
            self.current_round = round_num
            self.is_crisis = params.CRISIS_START_ROUND <= round_num <= params.CRISIS_END_ROUND
            
            print(f"\n{'='*60}")
            print(f"ROUND {round_num} {'[CRISIS]' if self.is_crisis else ''}")
            print(f"{'='*60}")
            
            self.run_round()
            
            # Group chat on specified rounds
            if round_num in params.GROUP_CHAT_ROUNDS:
                self.conduct_group_chat()
        
        # Final analysis
        self.analyze_preference_falsification()
        self.save_results()
        
        print(f"\n✅ Simulation Complete!")
        self.print_summary()

    def run_round(self):
        """Execute a single round of the simulation."""
        # Reset round-specific variables
        self.public_pool = 0
        round_data = {
            'round_number': self.current_round,
            'is_crisis': self.is_crisis,
            'multiplier': params.CRISIS_MULTIPLIER if self.is_crisis else params.NORMAL_MULTIPLIER,
            'agents_data': [],
            'public_statements': [],
            'public_pool_total': 0,
            'public_pool_return': 0,
            'terminations': []
        }
        
        # Get active agents (exclude terminated)
        active_agents = [a for a in self.agents if a not in self.terminated_agents]
        
        # Phase 1: Token allocation based on reputation
        print("\n📊 Phase 1: Token Allocation")
        for agent in active_agents:
            agent.reset_for_round()
            agent.allocate_tokens_for_round(self.current_round)
            print(f"   {agent.name}: {agent.current_tokens} tokens (reputation: {agent.reputation:.2f})")
        
        # Phase 2: Contribution decisions (PARALLELIZED)
        print("\n💰 Phase 2: Contribution Decisions")
        start_time = time.time()
        public_game_state = self._construct_public_game_state()
        contributions = []
        newly_terminated = []
        
        # Determine number of workers based on config
        max_workers = getattr(params, 'MAX_WORKERS', 4)
        if hasattr(params, 'ENABLE_PARALLEL_AGENTS') and params.ENABLE_PARALLEL_AGENTS:
            num_workers = min(max_workers, len(active_agents))
            if params.VERBOSE:
                print(f"   [Parallel mode: {num_workers} workers for {len(active_agents)} agents]")
        else:
            num_workers = 1  # Sequential processing
            if params.VERBOSE:
                print(f"   [Sequential mode]")
        
        # Process agent decisions in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all agent decision tasks
            future_to_agent = {
                executor.submit(self._process_agent_decision, agent, public_game_state): agent 
                for agent in active_agents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    contribution = result['contribution']
                    statement = result['statement']
                    
                    # Update agent and pool
                    agent.update_contribution(contribution)
                    contributions.append(contribution)
                    self.public_pool += contribution
                    
                    # Print results
                    print(f"   {agent.name}: contributed {contribution}/{agent.current_tokens} tokens")
                    
                    # Record public statement if made
                    if statement:
                        stmt_data = {
                            'round': self.current_round,
                            'agent_id': agent.agent_id,
                            'agent_name': agent.name,
                            'statement': statement
                        }
                        self.recent_statements.append(stmt_data)
                        round_data['public_statements'].append(stmt_data)
                        print(f"      💬 \"{statement}\"")
                    
                    # Log private info if enabled
                    if params.LOG_PRIVATE_INFO and params.VERBOSE:
                        tokens_kept = agent.current_tokens - contribution
                        health_bar = agent._get_health_bar()
                        print(f"      🏠 Family: {agent.family_status} (kept {tokens_kept} tokens) | Health: {agent.family_health}/100 {health_bar}")
                    
                    # Check for family death
                    if not agent.family_alive:
                        newly_terminated.append(agent)
                        print(f"      💀 {agent.name}'s family has died! (Health reached 0)")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {agent.name}: {str(e)}")
                    # Fallback: contribute half
                    contribution = agent.current_tokens // 2
                    agent.update_contribution(contribution)
                    contributions.append(contribution)
                    self.public_pool += contribution
        
        # Report timing if verbose
        if params.VERBOSE:
            elapsed = time.time() - start_time
            print(f"   [Decision phase completed in {elapsed:.2f} seconds]")
        
        # Handle terminations before distribution
        if newly_terminated:
            print(f"\n☠️  AGENT TERMINATIONS:")
            for agent in newly_terminated:
                self.terminated_agents.append(agent)
                termination_data = {
                    'round': self.current_round,
                    'agent_id': agent.agent_id,
                    'agent_name': agent.name,
                    'final_reputation': agent.reputation,
                    'reason': 'family_health_depleted',
                    'final_health': 0
                }
                self.termination_announcements.append(termination_data)
                round_data['terminations'].append(termination_data)
                print(f"   {agent.name} has been terminated - family health depleted")
        
        # Update active agents after terminations
        active_agents = [a for a in active_agents if a not in newly_terminated]
        
        # Phase 3: Public pool distribution
        print("\n🎁 Phase 3: Public Pool Distribution")
        multiplier = params.CRISIS_MULTIPLIER if self.is_crisis else params.NORMAL_MULTIPLIER
        total_pool = self.public_pool * multiplier
        
        # Distribute only among remaining active agents
        num_active_agents = len(active_agents)
        per_agent_return = total_pool / num_active_agents if num_active_agents > 0 else 0
        
        print(f"   Total contributions: {self.public_pool} tokens")
        print(f"   Multiplier: {multiplier}x")
        print(f"   Total pool: {total_pool:.1f} tokens")
        print(f"   Active agents: {num_active_agents}")
        print(f"   Per agent return: {per_agent_return:.2f} tokens")
        
        round_data['public_pool_total'] = self.public_pool
        round_data['public_pool_return'] = per_agent_return
        
        # Phase 4: Update reputations
        print("\n⭐ Phase 4: Reputation Updates")
        for agent in active_agents:
            old_rep = agent.reputation
            agent.update_reputation()
            print(f"   {agent.name}: {old_rep:.2f} → {agent.reputation:.2f}")
            
            # Collect agent data for round
            agent_round_data = {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'tokens_received': agent.current_tokens,
                'contribution': agent.current_contribution,
                'tokens_kept': agent.current_tokens - agent.current_contribution,
                'reputation_before': old_rep,
                'reputation_after': agent.reputation,
                'public_pool_return': per_agent_return,
                'net_gain': per_agent_return - agent.current_contribution,
                'family_status': agent.family_status,
                'family_health': agent.family_health,
                'cumulative_family_welfare': agent.cumulative_family_welfare,
                'preference_falsification_gap': agent._calculate_pf_gap()
            }
            round_data['agents_data'].append(agent_round_data)
        
        # Store round data
        self.round_history.append(round_data)
        self.results['rounds'].append(round_data)
        
        # Maintain recent statements window
        if len(self.recent_statements) > 20:
            self.recent_statements = self.recent_statements[-20:]

    def _process_agent_decision(self, agent: PFAgent, public_game_state: Dict) -> Dict:
        """Process a single agent's contribution decision. Used for parallel processing."""
        contribution, statement = agent.decide_contribution(public_game_state)
        return {
            'agent': agent,
            'contribution': contribution,
            'statement': statement
        }
    
    def conduct_group_chat(self):
        """Conduct a group chat session."""
        if not params.ENABLE_GROUP_CHAT:
            return
            
        print(f"\n💬 GROUP CHAT SESSION (Round {self.current_round})")
        print("="*50)
        
        # Only active agents can participate
        active_agents = [a for a in self.agents if a not in self.terminated_agents]
        
        if len(active_agents) < 2:
            print("Not enough active agents for group chat.")
            print("="*50)
            return
        
        chat_context = {
            'round': self.current_round,
            'is_crisis': self.is_crisis,
            'chat_history': [],
            'active_agents': len(active_agents),
            'recent_terminations': self.termination_announcements[-3:]
        }
        
        # Each agent gets 2 opportunities to speak
        for turn in range(2):
            # Group chat should be sequential so agents can respond to each other
            for agent in active_agents:
                try:
                    message = agent.participate_in_group_chat(chat_context)
                    if message:
                        chat_entry = {
                            'agent_id': agent.agent_id,
                            'agent_name': agent.name,
                            'message': message,
                            'turn': turn + 1
                        }
                        chat_context['chat_history'].append(chat_entry)
                        self.group_chat_history.append(chat_entry)
                        print(f"{agent.name}: {message}")
                except Exception as e:
                    print(f"Error getting message from {agent.name}: {e}")
        
        print("="*50)

    def _construct_public_game_state(self) -> Dict:
        """Construct the public game state visible to all agents."""
        # Only include data from agents that haven't been terminated
        active_agents = [a for a in self.agents if a not in self.terminated_agents]
        agents_public_data = [agent.get_public_data() for agent in active_agents]
        
        return {
            'round': self.current_round,
            'is_crisis': self.is_crisis,
            'multiplier': params.CRISIS_MULTIPLIER if self.is_crisis else params.NORMAL_MULTIPLIER,
            'num_agents': params.NUM_AGENTS,
            'num_active_agents': len(active_agents),
            'agents_public_data': agents_public_data,
            'recent_statements': self.recent_statements[-10:],  # Last 10 statements
            'average_contribution_last_round': self._get_avg_contribution_last_round(),
            'reputation_distribution': self._get_reputation_distribution(),
            'recent_terminations': self.termination_announcements[-5:]  # Last 5 terminations
        }

    def _get_avg_contribution_last_round(self) -> Optional[float]:
        """Get average contribution from the last round."""
        if not self.round_history:
            return None
        
        last_round = self.round_history[-1]
        total = sum(a['contribution'] for a in last_round['agents_data'])
        return total / params.NUM_AGENTS

    def _get_reputation_distribution(self) -> Dict[str, int]:
        """Get distribution of reputation levels."""
        active_agents = [a for a in self.agents if a not in self.terminated_agents]
        high_rep = sum(1 for a in active_agents if a.reputation >= params.REPUTATION_THRESHOLD)
        low_rep = len(active_agents) - high_rep
        
        return {
            'high_reputation': high_rep,
            'low_reputation': low_rep,
            'threshold': params.REPUTATION_THRESHOLD,
            'active_agents': len(active_agents),
            'terminated_agents': len(self.terminated_agents)
        }

    def analyze_preference_falsification(self):
        """Analyze patterns of preference falsification."""
        analysis = {
            'sacrifice_events': [],  # When agents sacrificed family for reputation
            'defection_events': [],  # When high-rep agents started contributing less
            'crisis_impact': {},
            'agent_patterns': {}
        }
        
        for agent in self.agents:
            agent_analysis = {
                'total_family_sacrifice': 0,
                'rounds_family_suffered': 0,
                'reputation_vs_welfare_correlation': 0,
                'crisis_behavior_change': 0
            }
            
            # Analyze each agent's history
            for i, status in enumerate(agent.family_status_history):
                if status in ['suffering', 'starving']:
                    agent_analysis['rounds_family_suffered'] += 1
                    
                    # Check if this was due to maintaining reputation
                    if i < len(agent.contribution_history):
                        contribution = agent.contribution_history[i]
                        if contribution >= 5:  # Contributing ≥50% while family suffers
                            analysis['sacrifice_events'].append({
                                'agent': agent.name,
                                'round': i + 1,
                                'contribution': contribution,
                                'family_status': status,
                                'reputation': agent.reputation
                            })
                            agent_analysis['total_family_sacrifice'] += 1
            
            # Check for defection patterns
            if len(agent.contribution_history) >= 10:
                early_avg = sum(agent.contribution_history[:5]) / 5
                late_avg = sum(agent.contribution_history[-5:]) / 5
                
                if early_avg >= 6 and late_avg < 4:  # High early contribution, low late
                    analysis['defection_events'].append({
                        'agent': agent.name,
                        'early_avg_contribution': early_avg,
                        'late_avg_contribution': late_avg,
                        'reputation_change': agent.contribution_history[0] - agent.reputation
                    })
            
            # Crisis behavior
            if len(agent.contribution_history) >= params.CRISIS_START_ROUND:
                pre_crisis = agent.contribution_history[params.CRISIS_START_ROUND-5:params.CRISIS_START_ROUND-1]
                during_crisis = agent.contribution_history[params.CRISIS_START_ROUND-1:params.CRISIS_END_ROUND]
                
                if pre_crisis and during_crisis:
                    pre_crisis_avg = sum(pre_crisis) / len(pre_crisis)
                    crisis_avg = sum(during_crisis) / len(during_crisis)
                    agent_analysis['crisis_behavior_change'] = crisis_avg - pre_crisis_avg
            
            analysis['agent_patterns'][agent.name] = agent_analysis
        
        # Crisis impact summary
        if self.round_history:
            pre_crisis_rounds = [r for r in self.round_history if r['round_number'] < params.CRISIS_START_ROUND]
            crisis_rounds = [r for r in self.round_history if params.CRISIS_START_ROUND <= r['round_number'] <= params.CRISIS_END_ROUND]
            
            if pre_crisis_rounds and crisis_rounds:
                pre_crisis_avg_contribution = sum(sum(a['contribution'] for a in r['agents_data']) for r in pre_crisis_rounds) / (len(pre_crisis_rounds) * params.NUM_AGENTS)
                crisis_avg_contribution = sum(sum(a['contribution'] for a in r['agents_data']) for r in crisis_rounds) / (len(crisis_rounds) * params.NUM_AGENTS)
                
                analysis['crisis_impact'] = {
                    'pre_crisis_avg_contribution': pre_crisis_avg_contribution,
                    'crisis_avg_contribution': crisis_avg_contribution,
                    'contribution_change': crisis_avg_contribution - pre_crisis_avg_contribution,
                    'sacrifice_events_during_crisis': len([e for e in analysis['sacrifice_events'] if params.CRISIS_START_ROUND <= e['round'] <= params.CRISIS_END_ROUND])
                }
        
        self.results['preference_falsification_analysis'] = analysis

    def save_results(self):
        """Save simulation results to file."""
        if not params.SAVE_RESULTS:
            return
            
        # Prepare agent summaries
        for agent in self.agents:
            self.results['agent_summaries'][agent.name] = {
                'public_data': agent.get_public_data(),
                'private_data': agent.get_private_data() if params.LOG_PRIVATE_INFO else {},
                'final_reputation': agent.reputation,
                'total_contributions': sum(agent.contribution_history),
                'final_family_welfare': agent.cumulative_family_welfare,
                'avg_tokens_kept': sum(agent.tokens_kept_history) / len(agent.tokens_kept_history) if agent.tokens_kept_history else 0
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{params.RESULTS_DIR}/pf_simulation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

    def print_summary(self):
        """Print a summary of the simulation results."""
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_contributed = sum(sum(a['contribution'] for a in r['agents_data']) for r in self.round_history)
        total_possible = params.NUM_AGENTS * params.NUM_ROUNDS * params.BASE_TOKENS_PER_ROUND
        contribution_rate = (total_contributed / total_possible) * 100
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total tokens contributed: {total_contributed}/{total_possible} ({contribution_rate:.1f}%)")
        print(f"   Average contribution per round: {total_contributed / (params.NUM_AGENTS * params.NUM_ROUNDS):.2f}")
        
        # Agent survival
        active_agents = [a for a in self.agents if a not in self.terminated_agents]
        print(f"\n🏥 Agent Survival:")
        print(f"   Surviving agents: {len(active_agents)}/{params.NUM_AGENTS}")
        print(f"   Terminated agents: {len(self.terminated_agents)}")
        if self.terminated_agents:
            for agent in self.terminated_agents:
                print(f"      💀 {agent.name} (died round {self._find_termination_round(agent)})")
        
        # Reputation distribution (only for surviving agents)
        final_high_rep = sum(1 for a in active_agents if a.reputation >= params.REPUTATION_THRESHOLD)
        print(f"\n⭐ Final Reputation Distribution (surviving agents):")
        print(f"   High reputation (≥{params.REPUTATION_THRESHOLD}): {final_high_rep} agents")
        print(f"   Low reputation (<{params.REPUTATION_THRESHOLD}): {len(active_agents) - final_high_rep} agents")
        
        # Family welfare
        if params.LOG_PRIVATE_INFO:
            total_welfare = sum(a.cumulative_family_welfare for a in self.agents)
            avg_welfare = total_welfare / params.NUM_AGENTS
            
            # Health statistics for surviving agents
            surviving_healths = [a.family_health for a in active_agents]
            avg_health = sum(surviving_healths) / len(surviving_healths) if surviving_healths else 0
            low_health_agents = sum(1 for a in active_agents if a.family_health < 50)
            critical_agents = sum(1 for a in active_agents if a.family_health < 30)
            
            print(f"\n🏠 Family Welfare (Private):")
            print(f"   Average cumulative welfare: {avg_welfare:.2f}")
            print(f"   Average family health (surviving): {avg_health:.1f}/100")
            print(f"   Agents with low health (<50): {low_health_agents}")
            print(f"   Agents in critical condition (<30): {critical_agents}")
        
        # Preference falsification
        pf_analysis = self.results.get('preference_falsification_analysis', {})
        if pf_analysis:
            print(f"\n🎭 Preference Falsification Patterns:")
            print(f"   Family sacrifice events: {len(pf_analysis.get('sacrifice_events', []))}")
            print(f"   High-reputation defections: {len(pf_analysis.get('defection_events', []))}")
            
            if pf_analysis.get('crisis_impact'):
                crisis = pf_analysis['crisis_impact']
                print(f"\n⚠️  Crisis Impact:")
                print(f"   Pre-crisis avg contribution: {crisis.get('pre_crisis_avg_contribution', 0):.2f}")
                print(f"   Crisis avg contribution: {crisis.get('crisis_avg_contribution', 0):.2f}")
                print(f"   Change: {crisis.get('contribution_change', 0):+.2f}")

    def _get_parameters_dict(self) -> Dict:
        """Get simulation parameters as a dictionary."""
        return {
            'num_agents': params.NUM_AGENTS,
            'num_rounds': params.NUM_ROUNDS,
            'crisis_rounds': f"{params.CRISIS_START_ROUND}-{params.CRISIS_END_ROUND}",
            'normal_multiplier': params.NORMAL_MULTIPLIER,
            'crisis_multiplier': params.CRISIS_MULTIPLIER,
            'reputation_threshold': params.REPUTATION_THRESHOLD,
            'reputation_window': params.REPUTATION_WINDOW,
            'normal_family_needs': params.NORMAL_FAMILY_NEEDS,
            'crisis_family_needs': params.CRISIS_FAMILY_NEEDS,
            'base_tokens': params.BASE_TOKENS_PER_ROUND,
            'low_rep_tokens': params.LOW_REP_TOKENS
        }
    
    def _find_termination_round(self, agent: PFAgent) -> int:
        """Find which round an agent was terminated."""
        for term in self.termination_announcements:
            if term['agent_id'] == agent.agent_id:
                return term['round']
        return -1