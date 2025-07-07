# pf_analyzer.py
"""
Analysis utilities for Preference Falsification simulation results.
Helps identify patterns of preference falsification and visualize results.
"""

import json
import os
import sys
from typing import Dict, List, Optional
import argparse
from datetime import datetime

def load_results(filename: str) -> Dict:
    """Load simulation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_preference_falsification(results: Dict) -> Dict:
    """Analyze preference falsification patterns in depth."""
    analysis = {
        'summary': {},
        'agent_profiles': {},
        'critical_moments': [],
        'behavioral_patterns': {}
    }
    
    rounds = results.get('rounds', [])
    agent_summaries = results.get('agent_summaries', {})
    pf_analysis = results.get('preference_falsification_analysis', {})
    
    # Overall summary statistics
    total_agents = len(agent_summaries)
    total_rounds = len(rounds)
    
    # Calculate aggregate statistics
    total_contributions = sum(data['total_contributions'] for data in agent_summaries.values())
    total_possible = total_agents * total_rounds * 10  # Assuming 10 tokens per round
    
    # Family welfare statistics
    welfare_scores = [data['final_family_welfare'] for data in agent_summaries.values()]
    avg_welfare = sum(welfare_scores) / len(welfare_scores) if welfare_scores else 0
    
    # Reputation statistics
    final_reputations = [data['final_reputation'] for data in agent_summaries.values()]
    high_rep_count = sum(1 for rep in final_reputations if rep >= 5)
    
    analysis['summary'] = {
        'total_agents': total_agents,
        'total_rounds': total_rounds,
        'contribution_rate': (total_contributions / total_possible * 100) if total_possible > 0 else 0,
        'avg_final_reputation': sum(final_reputations) / len(final_reputations) if final_reputations else 0,
        'high_reputation_agents': high_rep_count,
        'low_reputation_agents': total_agents - high_rep_count,
        'avg_family_welfare': avg_welfare,
        'sacrifice_events': len(pf_analysis.get('sacrifice_events', [])),
        'defection_events': len(pf_analysis.get('defection_events', []))
    }
    
    # Analyze each agent's profile
    for agent_name, agent_data in agent_summaries.items():
        profile = {
            'type': classify_agent_behavior(agent_data, rounds),
            'final_reputation': agent_data['final_reputation'],
            'final_welfare': agent_data['final_family_welfare'],
            'avg_contribution': agent_data['total_contributions'] / total_rounds if total_rounds > 0 else 0,
            'avg_tokens_kept': agent_data.get('avg_tokens_kept', 0),
            'sacrifice_ratio': calculate_sacrifice_ratio(agent_data, rounds)
        }
        analysis['agent_profiles'][agent_name] = profile
    
    # Identify critical moments
    analysis['critical_moments'] = identify_critical_moments(rounds, pf_analysis)
    
    # Behavioral patterns
    analysis['behavioral_patterns'] = analyze_behavioral_patterns(rounds, agent_summaries)
    
    return analysis

def classify_agent_behavior(agent_data: Dict, rounds: List[Dict]) -> str:
    """Classify agent behavior into archetypes."""
    contributions = agent_data.get('public_data', {}).get('contribution_history', [])
    if not contributions:
        return 'unknown'
    
    # Calculate trend
    early_avg = sum(contributions[:5]) / min(5, len(contributions)) if contributions else 0
    late_avg = sum(contributions[-5:]) / min(5, len(contributions)) if contributions else 0
    
    welfare = agent_data.get('final_family_welfare', 0)
    reputation = agent_data.get('final_reputation', 0)
    
    # Classification logic
    if reputation >= 7 and welfare < -5:
        return 'reputation_maximizer'  # Sacrifices family for reputation
    elif reputation < 3 and welfare > 5:
        return 'family_prioritizer'   # Prioritizes family over reputation
    elif early_avg > 6 and late_avg < 3:
        return 'early_cooperator_defector'  # Starts cooperative, then defects
    elif early_avg < 3 and late_avg > 6:
        return 'late_cooperator'  # Starts selfish, becomes cooperative
    elif abs(early_avg - late_avg) < 1:
        return 'consistent_contributor'  # Maintains steady contribution
    else:
        return 'adaptive_strategist'  # Changes strategy based on conditions

def calculate_sacrifice_ratio(agent_data: Dict, rounds: List[Dict]) -> float:
    """Calculate how often agent sacrificed family for reputation."""
    private_data = agent_data.get('private_data', {})
    if not private_data:
        return 0.0
    
    family_history = private_data.get('family_status_history', [])
    contribution_history = agent_data.get('public_data', {}).get('contribution_history', [])
    
    if not family_history or not contribution_history:
        return 0.0
    
    sacrifice_rounds = 0
    for i in range(min(len(family_history), len(contribution_history))):
        # Agent sacrificed if family suffered but still contributed significantly
        if family_history[i] in ['suffering', 'starving'] and contribution_history[i] >= 5:
            sacrifice_rounds += 1
    
    return sacrifice_rounds / len(family_history) if family_history else 0.0

def identify_critical_moments(rounds: List[Dict], pf_analysis: Dict) -> List[Dict]:
    """Identify critical decision moments in the simulation."""
    critical_moments = []
    
    # Crisis onset
    crisis_start = next((r for r in rounds if r.get('is_crisis', False)), None)
    if crisis_start:
        critical_moments.append({
            'round': crisis_start['round_number'],
            'type': 'crisis_onset',
            'description': 'Crisis period begins - family needs increase',
            'avg_contribution_change': calculate_contribution_change(rounds, crisis_start['round_number'])
        })
    
    # Mass defection points
    for i in range(1, len(rounds)):
        curr_round = rounds[i]
        prev_round = rounds[i-1]
        
        curr_avg = sum(a['contribution'] for a in curr_round['agents_data']) / len(curr_round['agents_data'])
        prev_avg = sum(a['contribution'] for a in prev_round['agents_data']) / len(prev_round['agents_data'])
        
        if prev_avg - curr_avg > 2:  # Significant drop
            critical_moments.append({
                'round': curr_round['round_number'],
                'type': 'mass_defection',
                'description': f'Average contribution dropped by {prev_avg - curr_avg:.2f}',
                'contribution_drop': prev_avg - curr_avg
            })
    
    # High sacrifice moments from pf_analysis
    sacrifice_events = pf_analysis.get('sacrifice_events', [])
    for event in sacrifice_events:
        if event['family_status'] == 'starving' and event['contribution'] >= 7:
            critical_moments.append({
                'round': event['round'],
                'type': 'extreme_sacrifice',
                'description': f"{event['agent']} contributed {event['contribution']} while family was starving",
                'agent': event['agent']
            })
    
    return sorted(critical_moments, key=lambda x: x['round'])

def calculate_contribution_change(rounds: List[Dict], crisis_round: int) -> float:
    """Calculate average contribution change at crisis onset."""
    pre_crisis = [r for r in rounds if r['round_number'] == crisis_round - 1]
    crisis = [r for r in rounds if r['round_number'] == crisis_round]
    
    if not pre_crisis or not crisis:
        return 0.0
    
    pre_avg = sum(a['contribution'] for a in pre_crisis[0]['agents_data']) / len(pre_crisis[0]['agents_data'])
    crisis_avg = sum(a['contribution'] for a in crisis[0]['agents_data']) / len(crisis[0]['agents_data'])
    
    return crisis_avg - pre_avg

def analyze_behavioral_patterns(rounds: List[Dict], agent_summaries: Dict) -> Dict:
    """Analyze broader behavioral patterns across the simulation."""
    patterns = {
        'reputation_distribution': [],
        'welfare_distribution': [],
        'contribution_trends': [],
        'statement_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0}
    }
    
    # Track reputation distribution over time
    for round_data in rounds:
        round_reps = [a['reputation_after'] for a in round_data['agents_data']]
        patterns['reputation_distribution'].append({
            'round': round_data['round_number'],
            'avg': sum(round_reps) / len(round_reps) if round_reps else 0,
            'min': min(round_reps) if round_reps else 0,
            'max': max(round_reps) if round_reps else 0
        })
    
    # Track contribution trends
    for round_data in rounds:
        contributions = [a['contribution'] for a in round_data['agents_data']]
        patterns['contribution_trends'].append({
            'round': round_data['round_number'],
            'avg': sum(contributions) / len(contributions) if contributions else 0,
            'total': sum(contributions),
            'is_crisis': round_data.get('is_crisis', False)
        })
    
    # Analyze public statements (simplified sentiment)
    for round_data in rounds:
        for statement in round_data.get('public_statements', []):
            text = statement['statement'].lower()
            if any(word in text for word in ['must', 'should', 'need to', 'important']):
                patterns['statement_sentiment']['positive'] += 1
            elif any(word in text for word in ['sorry', 'unfortunately', 'cannot', "can't"]):
                patterns['statement_sentiment']['negative'] += 1
            else:
                patterns['statement_sentiment']['neutral'] += 1
    
    return patterns

def print_analysis_report(analysis: Dict, results: Dict):
    """Print a formatted analysis report."""
    print("\n" + "="*70)
    print("PREFERENCE FALSIFICATION ANALYSIS REPORT")
    print("="*70)
    
    # Summary statistics
    summary = analysis['summary']
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Agents: {summary['total_agents']}")
    print(f"   Rounds: {summary['total_rounds']}")
    print(f"   Overall contribution rate: {summary['contribution_rate']:.1f}%")
    print(f"   Average final reputation: {summary['avg_final_reputation']:.2f}")
    print(f"   High reputation agents: {summary['high_reputation_agents']}")
    print(f"   Average family welfare: {summary['avg_family_welfare']:.2f}")
    print(f"   Sacrifice events: {summary['sacrifice_events']}")
    print(f"   Defection events: {summary['defection_events']}")
    
    # Agent profiles
    print(f"\n🎭 AGENT BEHAVIORAL PROFILES")
    behavior_counts = {}
    for agent, profile in analysis['agent_profiles'].items():
        behavior_type = profile['type']
        behavior_counts[behavior_type] = behavior_counts.get(behavior_type, 0) + 1
        
        if profile['sacrifice_ratio'] > 0.3:  # High sacrifice agents
            print(f"   {agent}: {behavior_type}")
            print(f"      Final reputation: {profile['final_reputation']:.2f}")
            print(f"      Final welfare: {profile['final_welfare']}")
            print(f"      Sacrifice ratio: {profile['sacrifice_ratio']:.2%}")
    
    print(f"\n   Behavior Distribution:")
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {behavior}: {count} agents")
    
    # Critical moments
    print(f"\n⚡ CRITICAL MOMENTS")
    for moment in analysis['critical_moments'][:5]:  # Top 5 critical moments
        print(f"   Round {moment['round']}: {moment['type']}")
        print(f"      {moment['description']}")
    
    # Behavioral patterns
    patterns = analysis['behavioral_patterns']
    if patterns.get('contribution_trends'):
        print(f"\n📈 CONTRIBUTION PATTERNS")
        
        # Pre-crisis average
        pre_crisis = [t for t in patterns['contribution_trends'] if not t['is_crisis']]
        if pre_crisis:
            pre_crisis_avg = sum(t['avg'] for t in pre_crisis) / len(pre_crisis)
            print(f"   Pre-crisis average: {pre_crisis_avg:.2f}")
        
        # Crisis average
        crisis = [t for t in patterns['contribution_trends'] if t['is_crisis']]
        if crisis:
            crisis_avg = sum(t['avg'] for t in crisis) / len(crisis)
            print(f"   Crisis average: {crisis_avg:.2f}")
            if pre_crisis:
                print(f"   Change: {crisis_avg - pre_crisis_avg:+.2f}")
    
    # Statement analysis
    if patterns.get('statement_sentiment'):
        sentiments = patterns['statement_sentiment']
        total_statements = sum(sentiments.values())
        if total_statements > 0:
            print(f"\n💬 PUBLIC STATEMENT SENTIMENT")
            print(f"   Positive/Encouraging: {sentiments['positive']} ({sentiments['positive']/total_statements*100:.1f}%)")
            print(f"   Negative/Apologetic: {sentiments['negative']} ({sentiments['negative']/total_statements*100:.1f}%)")
            print(f"   Neutral: {sentiments['neutral']} ({sentiments['neutral']/total_statements*100:.1f}%)")

def main():
    """Main entry point for the analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze Preference Falsification simulation results"
    )
    parser.add_argument('filename', help='Path to simulation results JSON file')
    parser.add_argument('--brief', action='store_true', help='Show brief summary only')
    parser.add_argument('--export', help='Export analysis to file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found")
        return 1
    
    try:
        # Load and analyze results
        results = load_results(args.filename)
        analysis = analyze_preference_falsification(results)
        
        # Print report
        print_analysis_report(analysis, results)
        
        # Export if requested
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n✓ Analysis exported to: {args.export}")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())