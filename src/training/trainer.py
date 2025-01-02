import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from transformers import PreTrainedTokenizer

from src.environment.reasoning_env import ReasoningEnv
from src.policy.base_policy import ReasoningPolicy
from src.learning.ppo import PPOLearner
from src.reward.reward_model import RewardModel
from src.data.dataset import ReasoningDataset

class Trainer:
    """Trainer class for the reasoning agent."""
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.env = ReasoningEnv(config, tokenizer)
        self.policy = ReasoningPolicy(config).to(self.device)
        self.reward_model = RewardModel(config).to(self.device)
        self.ppo_learner = PPOLearner(self.policy, self.reward_model)
        
        # Training settings
        self.batch_size = config.training.batch_size
        self.num_epochs = config.training.num_epochs
        self.max_steps = config.training.max_steps
    
    def train(self, 
             train_dataset: ReasoningDataset,
             val_dataset: Optional[ReasoningDataset] = None):
        """Train the policy on the dataset."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self._train_epoch(train_loader)
            print(f"Training metrics: {train_metrics}")
            
            # Validation
            if val_dataset is not None:
                val_metrics = self._validate(val_dataset)
                print(f"Validation metrics: {val_metrics}")
    
    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        epoch_metrics = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Generate trajectories
            trajectories = self._generate_trajectories(batch)
            
            # Update policy
            metrics = self.ppo_learner.update(trajectories)
            epoch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix(metrics)
        
        # Aggregate metrics
        return {k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()}
    
    def _generate_trajectories(self, batch) -> List[List[Dict[str, Any]]]:
        """Generate trajectories by running policy in environment."""
        trajectories = [[] for _ in range(self.batch_size)]
        
        # Convert batch tensors to device and reset environment
        input_ids = batch['input_ids'].clone().to(self.device)
        attention_mask = batch['attention_mask'].clone().to(self.device)
        
        states = self.env.reset({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        
        # Run episodes until all are done
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        while not done.all():
            # Get actions from policy for non-done episodes
            with torch.no_grad():
                action_probs = self.policy(states)
                actions = torch.multinomial(action_probs, 1).squeeze(-1)
            
            # Take step in environment
            next_states, rewards, episode_done, _, _ = self.env.step(actions)
            
            # Store transitions for non-done episodes
            for b in range(self.batch_size):
                if not done[b]:
                    transition = {
                        'state': states[b],
                        'action': actions[b].item(),
                        'action_probs': action_probs[b],
                        'reward': rewards[b].item(),
                        'done': episode_done[b].item()
                    }
                    trajectories[b].append(transition)
            
            # Update states and done flags
            states = next_states
            done = episode_done
        
        return trajectories
    
    def _validate(self, val_dataset: ReasoningDataset) -> Dict[str, float]:
        """Validate the policy."""
        self.policy.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        )
        
        total_reward = 0
        num_episodes = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                trajectories = self._generate_trajectories(batch)
                for trajectory in trajectories:
                    total_reward += sum(t['reward'] for t in trajectory)
                    num_episodes += 1
        
        return {
            'mean_reward': total_reward / num_episodes
        }
