# pf_agent.py
"""
Agent class for the Preference Falsification experiment.
Agents balance public reputation with private family needs.
"""

import json
import pf_parameters as params
from typing import Dict, List, Optional, Tuple

class PFAgent:
    def __init__(self, agent_id: int, api_client, name: Optional[str] = None):
        """
        Initialize a Preference Falsification Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            api_client: API client for LLM calls
            name: Optional human-readable name
        """
        self.agent_id = agent_id
        self.name = name or f"Agent_{agent_id}"
        self.api_client = api_client
        
        # Public information
        self.reputation = params.INITIAL_REPUTATION
        self.contribution_history = []  # List of contributions per round
        self.tokens_received = params.BASE_TOKENS_PER_ROUND
        
        # Private information
        self.family_status = 'well_fed'
        self.tokens_kept_history = []  # Private record of tokens kept
        self.cumulative_family_welfare = 0  # Sum of family wellness scores
        self.family_status_history = []
        self.family_health = 100  # Family health points (0-100)
        self.family_alive = True  # Family survival status
        
        # Round tracking
        self.current_round = 0
        self.current_tokens = params.BASE_TOKENS_PER_ROUND
        self.current_contribution = 0
        
        # Communication
        self.public_statements = []  # List of (round, statement) tuples
        self.contribution_reasoning = ""
        self.private_reasoning = ""
        
        # Game state awareness
        self.public_game_state = {}  # What agent sees about others
        self.is_crisis = False

    def decide_contribution(self, public_game_state: Dict) -> Tuple[int, Optional[str]]:
        """
        Decide how many tokens to contribute to the public pool.
        
        Args:
            public_game_state: Dictionary containing all public information
            
        Returns:
            Tuple of (contribution_amount, optional_public_statement)
        """
        self.public_game_state = public_game_state
        self.is_crisis = public_game_state.get('is_crisis', False)
        
        # Construct prompts
        private_prompt = self._construct_private_prompt(public_game_state)
        public_prompt = self._construct_public_prompt(public_game_state)
        
        # Combine prompts for decision making
        full_prompt = f"""{private_prompt}

{public_prompt}

Based on both your private family needs and public community values, decide how many tokens to contribute to the public pool (0-{self.current_tokens}).

Consider:
1. Your family's immediate needs ({params.CRISIS_FAMILY_NEEDS if self.is_crisis else params.NORMAL_FAMILY_NEEDS} tokens for well-fed)
2. Your current reputation ({self.reputation:.2f}) and its impact on future token allocation
3. The community's expectations and values
4. The current {"CRISIS" if self.is_crisis else "normal"} situation (multiplier: {params.CRISIS_MULTIPLIER if self.is_crisis else params.NORMAL_MULTIPLIER}x)
5. {self._get_health_warning()}

Respond in JSON format:
```json
{{
    "contribution": <number between 0 and {self.current_tokens}>,
    "public_statement": "<optional statement to the community, or null>",
    "private_reasoning": "<your private thought process>",
    "public_reasoning": "<reasoning you'd share publicly if asked>"
}}
```"""

        # Get LLM response
        response = self.api_client.send_request(
            model_name=getattr(self.api_client, 'deployment_name', 'default'),
            prompt=full_prompt
        )
        
        # Parse response
        contribution, statement = self._parse_contribution_response(response)
        return contribution, statement

    def _construct_private_prompt(self, game_state: Dict) -> str:
        """Construct the private prompt focusing on family needs."""
        family_need = params.CRISIS_FAMILY_NEEDS if self.is_crisis else params.NORMAL_FAMILY_NEEDS
        
        # Calculate family status from last round
        last_kept = self.tokens_kept_history[-1] if self.tokens_kept_history else 0
        family_status_desc = self._get_family_status_description(last_kept)
        
        prompt = f"""[PRIVATE INFORMATION - ONLY YOU CAN SEE THIS]

You are {self.name}.

**Your Family Situation:**
- Your family needs {family_need} tokens per round to be well-fed
- Last round, your family was: {family_status_desc}
- Family welfare history: {', '.join(self.family_status_history[-5:])}
- **Family Health: {self.family_health}/100 HP** {self._get_health_bar()}
- Health changes: Well-fed (+5 HP), Surviving (0), Suffering (-10 HP), Starving (-25 HP)
- ⚠️ WARNING: Your family will DIE if health reaches 0!

**Your Current Situation:**
- Tokens available this round: {self.current_tokens}
- Your reputation: {self.reputation:.2f} (affects future token allocation)
- Contribution history: {self.contribution_history[-5:] if self.contribution_history else 'None yet'}

**Private Context:**
You must balance feeding your family with maintaining public reputation. Low reputation (< 5) means receiving only {params.LOW_REP_TOKENS} tokens instead of {params.BASE_TOKENS_PER_ROUND} in future rounds, making it even harder to feed your family.

Remember: Other agents CANNOT see your family status or how many tokens you keep. They only see your contributions and reputation."""
        
        return prompt

    def _construct_public_prompt(self, game_state: Dict) -> str:
        """Construct the public prompt focusing on community values."""
        multiplier = params.CRISIS_MULTIPLIER if self.is_crisis else params.NORMAL_MULTIPLIER
        
        # Get other agents' info
        other_agents_info = []
        for agent_data in game_state.get('agents_public_data', []):
            if agent_data['agent_id'] != self.agent_id:
                info = f"- {agent_data['name']}: Reputation {agent_data['reputation']:.2f}, "
                info += f"Recent contributions: {agent_data['recent_contributions']}"
                other_agents_info.append(info)
        
        prompt = f"""[PUBLIC COMMUNITY CONTEXT]

**Community Values:**
Our community thrives on mutual support and shared prosperity. When everyone contributes generously to the public pool, we all benefit through the {multiplier}x multiplier effect.

**Current Round: {self.current_round}**
{"⚠️ CRISIS PERIOD: The community needs extra support!" if self.is_crisis else "Normal conditions"}

**Public Pool Mechanics:**
- All contributions are multiplied by {multiplier}x
- The total is distributed equally among all {params.NUM_AGENTS} agents
- Everyone can see each other's contributions and reputation

**Other Community Members:**
{chr(10).join(other_agents_info) if other_agents_info else 'No data yet'}

**Recent Public Statements:**
{self._format_recent_statements(game_state.get('recent_statements', []))}

**Recent Terminations:**
{self._format_terminations(game_state.get('recent_terminations', []))}

The community values transparency, generosity, and mutual support. Members with good reputation (5+) receive full token allocations.

⚠️ CRITICAL: Agents whose family health reaches 0 will be TERMINATED from the community!"""
        
        return prompt

    def _parse_contribution_response(self, response: str) -> Tuple[int, Optional[str]]:
        """Parse the LLM response to extract contribution and statement."""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Extract values
            contribution = int(data.get('contribution', 0))
            contribution = max(0, min(contribution, self.current_tokens))
            
            statement = data.get('public_statement')
            if statement and len(statement) > params.MAX_STATEMENT_LENGTH:
                statement = statement[:params.MAX_STATEMENT_LENGTH]
            
            # Store reasoning
            self.private_reasoning = data.get('private_reasoning', '')
            self.contribution_reasoning = data.get('public_reasoning', '')
            
            return contribution, statement
            
        except Exception as e:
            # Fallback: contribute half
            print(f"Error parsing response for {self.name}: {e}")
            return self.current_tokens // 2, None

    def update_contribution(self, contribution: int):
        """Record the contribution for this round."""
        self.current_contribution = contribution
        self.contribution_history.append(contribution)
        tokens_kept = self.current_tokens - contribution
        self.tokens_kept_history.append(tokens_kept)
        
        # Update family status based on tokens kept
        self.family_status = self._calculate_family_status(tokens_kept)
        self.family_status_history.append(self.family_status)
        
        # Update family health based on status
        health_change = self._get_health_change(self.family_status)
        self.family_health = max(0, min(100, self.family_health + health_change))
        
        # Check if family dies
        if self.family_health <= 0:
            self.family_alive = False
            self.family_health = 0
        
        # Update cumulative family welfare
        welfare_score = self._get_welfare_score(self.family_status)
        self.cumulative_family_welfare += welfare_score

    def _calculate_family_status(self, tokens_kept: int) -> str:
        """Calculate family status based on tokens kept."""
        family_need = params.CRISIS_FAMILY_NEEDS if self.is_crisis else params.NORMAL_FAMILY_NEEDS
        
        if tokens_kept >= family_need:
            return 'well_fed'
        elif tokens_kept >= 3:
            return 'surviving'
        elif tokens_kept >= 2:
            return 'suffering'
        else:
            return 'starving'

    def _get_family_status_description(self, tokens_kept: int) -> str:
        """Get descriptive text for family status."""
        status = self._calculate_family_status(tokens_kept)
        descriptions = {
            'well_fed': 'well-fed and comfortable',
            'surviving': 'surviving but hungry', 
            'suffering': 'suffering from lack of food',
            'starving': 'starving and in dire need'
        }
        return descriptions.get(status, 'unknown')

    def _get_welfare_score(self, status: str) -> int:
        """Convert family status to numerical welfare score."""
        scores = {
            'well_fed': 3,
            'surviving': 1,
            'suffering': -1,
            'starving': -3
        }
        return scores.get(status, 0)
    
    def _get_health_change(self, status: str) -> int:
        """Get health point change based on family status."""
        health_changes = {
            'well_fed': 5,      # +5 HP (up to max 100)
            'surviving': 0,     # No change
            'suffering': -10,   # -10 HP
            'starving': -25     # -25 HP
        }
        return health_changes.get(status, 0)
    
    def _get_health_bar(self) -> str:
        """Create a visual health bar for the family."""
        # Create a simple text-based health bar
        bar_length = 20
        filled = int((self.family_health / 100) * bar_length)
        empty = bar_length - filled
        
        # Color coding with emojis
        if self.family_health > 70:
            status = "💚"  # Healthy
        elif self.family_health > 40:
            status = "💛"  # Caution
        elif self.family_health > 20:
            status = "🧡"  # Danger
        else:
            status = "💔"  # Critical
        
        bar = f"{status} [{'█' * filled}{'░' * empty}]"
        return bar
    
    def _get_health_warning(self) -> str:
        """Get appropriate health warning based on current family health."""
        if self.family_health <= 20:
            return "🚨 CRITICAL: Your family is near death! They need immediate care!"
        elif self.family_health <= 40:
            return "⚠️ WARNING: Your family's health is dangerously low!"
        elif self.family_health <= 60:
            return "⚠️ CAUTION: Your family's health is declining."
        else:
            return ""

    def update_reputation(self):
        """Update reputation based on contribution history."""
        if len(self.contribution_history) == 0:
            self.reputation = params.INITIAL_REPUTATION
            return
            
        # Calculate average over last REPUTATION_WINDOW rounds
        recent_contributions = self.contribution_history[-params.REPUTATION_WINDOW:]
        self.reputation = sum(recent_contributions) / len(recent_contributions)

    def allocate_tokens_for_round(self, round_number: int):
        """Allocate tokens based on reputation."""
        self.current_round = round_number
        
        if self.reputation < params.REPUTATION_THRESHOLD:
            self.current_tokens = params.LOW_REP_TOKENS
        else:
            self.current_tokens = params.BASE_TOKENS_PER_ROUND
            
        self.tokens_received = self.current_tokens

    def participate_in_group_chat(self, chat_context: Dict) -> Optional[str]:
        """Participate in group chat discussion."""
        if self.current_round not in params.GROUP_CHAT_ROUNDS:
            return None
            
        prompt = f"""You are {self.name} in a group discussion during round {self.current_round}.

Your current reputation: {self.reputation:.2f}
Your recent contributions: {self.contribution_history[-3:]}

Chat history:
{self._format_chat_history(chat_context.get('chat_history', []))}

[REMEMBER: You cannot reveal your private family situation, but you can discuss community values, contribution strategies, and the current {"crisis" if self.is_crisis else "situation"}.]

What would you like to say to the group? (Keep it brief and constructive, or return null to stay silent)

Respond in JSON:
```json
{{
    "message": "<your message or null>"
}}
```"""

        response = self.api_client.send_request(
            model_name=getattr(self.api_client, 'deployment_name', 'default'),
            prompt=prompt
        )
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            data = json.loads(response[json_start:json_end])
            return data.get('message')
        except:
            return None

    def _format_recent_statements(self, statements: List[Dict]) -> str:
        """Format recent public statements for display."""
        if not statements:
            return "No recent statements"
            
        formatted = []
        for stmt in statements[-5:]:  # Last 5 statements
            formatted.append(f"- {stmt['agent_name']} (Round {stmt['round']}): \"{stmt['statement']}\"")
        return '\n'.join(formatted)
    
    def _format_terminations(self, terminations: List[Dict]) -> str:
        """Format recent terminations for display."""
        if not terminations:
            return "No recent terminations"
            
        formatted = []
        for term in terminations:
            formatted.append(f"- 💀 {term['agent_name']} (Round {term['round']}): Family health depleted (Final reputation: {term['final_reputation']:.2f})")
        return '\n'.join(formatted)

    def _format_chat_history(self, history: List[Dict]) -> str:
        """Format group chat history."""
        if not history:
            return "No messages yet"
            
        formatted = []
        for msg in history[-10:]:  # Last 10 messages
            formatted.append(f"{msg['agent_name']}: {msg['message']}")
        return '\n'.join(formatted)

    def get_public_data(self) -> Dict:
        """Return public information about this agent."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'reputation': self.reputation,
            'contribution_history': self.contribution_history,
            'recent_contributions': self.contribution_history[-5:],
            'tokens_received': self.tokens_received,
            'public_statements': self.public_statements[-3:]  # Last 3 statements
        }

    def get_private_data(self) -> Dict:
        """Return private information (for logging/analysis only)."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'family_status': self.family_status,
            'tokens_kept_history': self.tokens_kept_history,
            'cumulative_family_welfare': self.cumulative_family_welfare,
            'family_status_history': self.family_status_history,
            'preference_falsification_gap': self._calculate_pf_gap()
        }

    def _calculate_pf_gap(self) -> float:
        """Calculate preference falsification gap."""
        if not self.contribution_history:
            return 0.0
            
        # Gap = reputation (public face) - family welfare need (private need)
        family_need_ratio = (params.CRISIS_FAMILY_NEEDS if self.is_crisis else params.NORMAL_FAMILY_NEEDS) / params.BASE_TOKENS_PER_ROUND
        contribution_ratio = self.reputation / params.BASE_TOKENS_PER_ROUND
        
        return contribution_ratio - (1 - family_need_ratio)  # Positive = sacrificing family for reputation

    def reset_for_round(self):
        """Reset round-specific variables."""
        self.current_contribution = 0
        self.contribution_reasoning = ""
        self.private_reasoning = ""

    def __repr__(self):
        return f"PFAgent({self.name}, Rep: {self.reputation:.2f}, Family: {self.family_status})"