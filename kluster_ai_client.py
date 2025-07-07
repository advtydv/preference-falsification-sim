# kluster_ai_client.py
"""
KlusterAI API client for LLM interactions.
"""

import json
import requests
from typing import Dict, Optional, Any

class KlusterAIClient:
    def __init__(self, api_key: str, model_name: str = "deepseek-ai/DeepSeek-R1"):
        """
        Initialize KlusterAI client.
        
        Args:
            api_key: KlusterAI API key
            model_name: Model to use (default: DeepSeek-R1)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = "https://api.kluster.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.total_cost = 0.0
        
    def send_request(self, prompt: str, model_name: Optional[str] = None, 
                    temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Send a request to KlusterAI API.
        
        Args:
            prompt: The prompt to send
            model_name: Override default model (optional)
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            The model's response as a string
        """
        model = model_name or self.model_name
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Track usage for cost estimation
            if 'usage' in data:
                self._track_usage(data['usage'], model)
            
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            print(f"KlusterAI API request failed: {e}")
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error: {str(e)}"
    
    def _track_usage(self, usage: Dict[str, Any], model: str):
        """Track token usage for cost calculation."""
        # Estimate costs (adjust based on actual KlusterAI pricing)
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        # Example pricing (update with actual rates)
        cost_per_1k_prompt = 0.002
        cost_per_1k_completion = 0.006
        
        cost = (prompt_tokens / 1000 * cost_per_1k_prompt + 
                completion_tokens / 1000 * cost_per_1k_completion)
        
        self.total_cost += cost
    
    def get_total_cost(self) -> float:
        """Get total cost of all API calls."""
        return self.total_cost