import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

class PPOLearner:
    """PPO implementation for training the policy."""
    
    def __init__(self,
                 policy: nn.Module,
                 reward_model: nn.Module,
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.policy = policy
        self.reward_model = reward_model
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def update(self, trajectories: List[List[Dict]]) -> Dict[str, float]:
        """Update policy using PPO algorithm.
        
        Args:
            trajectories: List of trajectories, where each trajectory is a list of
                        state-action-reward dictionaries.
                        
        Returns:
            Dictionary of training metrics.
        """
        # Combine all trajectories into batches
        states = []
        actions = []
        old_probs = []
        rewards = []
        dones = []
        
        # Process each trajectory
        max_length = max(len(traj) for traj in trajectories)
        batch_size = len(trajectories)
        
        # Initialize tensors
        device = next(self.policy.parameters()).device
        states = torch.zeros(batch_size, max_length, *trajectories[0][0]['state'].shape, device=device)
        actions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        old_probs = torch.zeros(batch_size, max_length, self.policy.config.action_dim, device=device)
        rewards = torch.zeros(batch_size, max_length, device=device)
        dones = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        masks = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        
        # Fill tensors
        for b, trajectory in enumerate(trajectories):
            length = len(trajectory)
            for t, transition in enumerate(trajectory):
                states[b, t] = transition['state']
                actions[b, t] = transition['action']
                old_probs[b, t] = transition['action_probs']
                rewards[b, t] = transition['reward']
                dones[b, t] = transition['done']
                masks[b, t] = True
        
        # Compute advantages
        with torch.no_grad():
            values = self.reward_model(states)
            next_values = torch.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]
            
            # GAE calculation
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            for t in reversed(range(max_length)):
                if t == max_length - 1:
                    next_val = 0
                else:
                    next_val = values[:, t + 1]
                delta = rewards[:, t] + (1 - dones[:, t].float()) * next_val - values[:, t]
                advantages[:, t] = delta + (1 - dones[:, t].float()) * self.value_coef * last_gae
                last_gae = advantages[:, t]
        
        # Normalize advantages
        advantages = (advantages - advantages[masks].mean()) / (advantages[masks].std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_entropy_loss = 0
        
        for _ in range(5):  # Multiple epochs
            # Compute policy loss for all timesteps at once
            action_probs = self.policy(states[masks])
            actions_taken = actions[masks]
            
            # Get probabilities of taken actions
            curr_probs = torch.gather(action_probs, 1, actions_taken.unsqueeze(-1)).squeeze(-1)
            old_action_probs = torch.gather(old_probs[masks], 1, actions_taken.unsqueeze(-1)).squeeze(-1)
            
            # Compute ratio and clipped ratio
            ratio = curr_probs / old_action_probs
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # Compute losses
            policy_loss = -torch.min(
                ratio * advantages[masks],
                clipped_ratio * advantages[masks]
            ).mean()
            
            entropy_loss = -torch.mean(
                torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
            )
            
            # Total loss
            loss = policy_loss - self.entropy_coef * entropy_loss
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / 5,
            'entropy': total_entropy_loss / 5,
            'mean_reward': rewards[masks].mean().item()
        }
