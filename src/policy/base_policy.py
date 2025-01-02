from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple

class BasePolicy(nn.Module):
    """Base class for all policies."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy network."""
        raise NotImplementedError
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        with torch.no_grad():
            action_probs = self.forward(state)
            actions = torch.multinomial(action_probs, 1)
            log_probs = torch.log(torch.gather(action_probs, 1, actions))
        return actions.squeeze(-1), log_probs.squeeze(-1)
    
    def save(self, path):
        """Save policy to disk."""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load policy from disk."""
        self.load_state_dict(torch.load(path))

class ReasoningPolicy(BasePolicy):
    """Implementation of the reasoning policy."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.hidden_size,
            nhead=config.model.num_heads,
            dim_feedforward=config.model.hidden_size * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_layers,
            norm=nn.LayerNorm(config.model.hidden_size)
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the policy network.
        
        Args:
            state: Tensor of shape (batch_size, seq_length, hidden_size)
            
        Returns:
            action_probs: Tensor of shape (batch_size, action_dim)
        """
        # Create attention mask (all ones for now)
        batch_size, seq_length = state.size()[:2]
        attention_mask = torch.ones(batch_size, seq_length, device=state.device)
        
        # Encode sequence
        encoded = self.encoder(state, src_key_padding_mask=attention_mask)
        
        # Pool sequence to single vector (mean pooling)
        pooled = encoded.mean(dim=1)
        
        # Get action probabilities
        action_probs = self.action_head(pooled)
        
        return action_probs
