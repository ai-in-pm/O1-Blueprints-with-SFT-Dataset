import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MCTSConfig:
    num_simulations: int = 50
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0

class MCTSNode:
    def __init__(self, prior: float, state):
        self.state = state
        self.prior = prior
        self.children: Dict = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    """Monte Carlo Tree Search implementation for reasoning tasks."""
    
    def __init__(self, policy, config: MCTSConfig):
        self.policy = policy
        self.config = config
        
    def search(self, root_state) -> Tuple[torch.Tensor, List[float]]:
        """Perform MCTS search starting from root state."""
        root = MCTSNode(prior=1.0, state=root_state)
        
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded:
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            value = self._expand_and_evaluate(node)
            
            # Backup
            self._backup(search_path, value)
        
        # Return action probabilities
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        action_probs = self._normalize_probs(visit_counts)
        
        return action_probs, [child.value for child in root.children.values()]
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child node using PUCT algorithm."""
        puct_scores = {}
        sqrt_total = np.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            q_value = -child.value  # Negative because of alternating perspective
            u_value = (self.config.c_puct * child.prior * sqrt_total / 
                      (1 + child.visit_count))
            puct_scores[action] = q_value + u_value
        
        action = max(puct_scores.items(), key=lambda x: x[1])[0]
        return action, node.children[action]
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node and return value estimate."""
        action_probs, value = self.policy(node.state)
        
        # Add Dirichlet noise to prior probabilities at root
        if node.visit_count == 0:
            action_probs = self._add_dirichlet_noise(action_probs)
        
        # Create children nodes
        for action, prob in enumerate(action_probs):
            if prob > 0:
                next_state = self._get_next_state(node.state, action)
                node.children[action] = MCTSNode(prior=prob, state=next_state)
        
        node.is_expanded = True
        return value
    
    def _backup(self, search_path: List[MCTSNode], value: float):
        """Backup value through search path."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip value for alternating perspective
    
    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature and normalize probabilities."""
        if self.config.temperature == 0:
            max_idx = np.argmax(probs)
            normalized = np.zeros_like(probs)
            normalized[max_idx] = 1.0
            return normalized
        
        probs = probs ** (1.0 / self.config.temperature)
        return probs / np.sum(probs)
    
    def _add_dirichlet_noise(self, probs: np.ndarray) -> np.ndarray:
        """Add Dirichlet noise to prior probabilities."""
        alpha = [self.config.dirichlet_alpha] * len(probs)
        noise = np.random.dirichlet(alpha)
        return ((1 - self.config.dirichlet_epsilon) * probs + 
                self.config.dirichlet_epsilon * noise)
    
    def _get_next_state(self, state, action):
        """Get next state after applying action."""
        # This should be implemented based on your environment
        raise NotImplementedError
