import gymnasium as gym
import torch
import numpy as np
from typing import Dict, Any, Tuple
from transformers import PreTrainedTokenizer

class ReasoningEnv(gym.Env):
    """Custom environment for reasoning tasks."""
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_steps = config.training.max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(config.action_dim)
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(config.model.max_seq_length, config.model.hidden_size),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_input = None
        self.current_step = None
        self.done = None
        self.batch_size = None
    
    def reset(self, input_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reset environment with new input."""
        self.current_input = {
            k: v.to(self.device) for k, v in input_data.items()
        }
        self.batch_size = self.current_input['input_ids'].size(0)
        self.current_step = torch.zeros(self.batch_size, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        return self._get_observation()
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            actions: Tensor of shape (batch_size,) containing actions for each batch item
            
        Returns:
            Tuple of (next_obs, rewards, dones, truncated, info)
        """
        if self.current_input is None:
            raise RuntimeError("Environment not initialized, call reset() first")
        
        # Update step counter for non-done environments
        self.current_step[~self.done] += 1
        
        # Check which environments are done
        self.done = self.current_step >= self.max_steps
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Compute rewards (placeholder for now)
        rewards = torch.zeros(self.batch_size, device=self.device)
        
        # Return step information
        return next_obs, rewards, self.done, False, {}
    
    def _get_observation(self) -> torch.Tensor:
        """Get current observation."""
        # Get input tensors
        input_ids = self.current_input['input_ids']
        attention_mask = self.current_input['attention_mask']
        
        # Create embeddings based on current step
        seq_length = input_ids.size(1)
        hidden_size = self.config.model.hidden_size
        
        # Create position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.float32, device=self.device)
        position_embeddings = position_ids.unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, hidden_size)
        
        # Apply attention mask and step-based modification
        step_scale = (self.current_step / self.max_steps).view(-1, 1, 1)
        position_embeddings = position_embeddings * attention_mask.unsqueeze(-1).float()
        position_embeddings = position_embeddings * (1.0 + step_scale)  # Scale embeddings based on step
        
        return position_embeddings
