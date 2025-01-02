import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class RewardModel(nn.Module):
    """Reward model for evaluating agent behaviors and outcomes."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Outcome evaluation network
        self.outcome_network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Process evaluation network
        self.process_network = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def compute_outcome_reward(self, final_state: torch.Tensor) -> torch.Tensor:
        """Compute reward based on the final outcome."""
        return self.outcome_network(final_state)
    
    def compute_process_reward(self, 
                             state: torch.Tensor,
                             action: torch.Tensor,
                             next_state: torch.Tensor) -> torch.Tensor:
        """Compute reward based on the reasoning process."""
        process_input = torch.cat([state, action], dim=-1)
        return self.process_network(process_input)
    
    def compute_total_reward(self,
                           state: torch.Tensor,
                           action: torch.Tensor,
                           next_state: torch.Tensor,
                           is_final: bool) -> torch.Tensor:
        """Compute combined reward incorporating both outcome and process."""
        process_reward = self.compute_process_reward(state, action, next_state)
        
        if is_final:
            outcome_reward = self.compute_outcome_reward(next_state)
            total_reward = (self.config.outcome_weight * outcome_reward +
                          self.config.process_weight * process_reward)
        else:
            total_reward = process_reward
            
        return total_reward
    
    def update_from_preferences(self,
                              preferred_trajectories: List[Tuple[torch.Tensor, ...]],
                              non_preferred_trajectories: List[Tuple[torch.Tensor, ...]]):
        """Update reward model using human preference data."""
        loss = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        for preferred, non_preferred in zip(preferred_trajectories, non_preferred_trajectories):
            preferred_reward = self.compute_total_reward(*preferred)
            non_preferred_reward = self.compute_total_reward(*non_preferred)
            
            # Preference loss using sigmoid
            loss += -torch.log(torch.sigmoid(preferred_reward - non_preferred_reward))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
