from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1
    max_seq_length: int = 512
    vocab_size: int = 50257  # GPT-2 vocabulary size

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    max_steps: int = 100  # Maximum steps per episode
    
@dataclass
class RewardConfig:
    outcome_weight: float = 0.7
    process_weight: float = 0.3
    preference_batch_size: int = 16
    preference_buffer_size: int = 10000
    
@dataclass
class SearchConfig:
    num_simulations: int = 100
    c_puct: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    reward: RewardConfig = RewardConfig()
    search: SearchConfig = SearchConfig()
    
    # Environment settings
    state_dim: int = 768
    action_dim: int = 100
    
    # PPO settings
    ppo_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Paths
    model_path: Optional[str] = None
    data_path: Optional[str] = None
    log_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.model.hidden_size > 0
        assert self.model.num_heads > 0
        assert self.model.num_layers > 0
        assert 0 <= self.model.dropout <= 1
        assert self.training.learning_rate > 0
        assert self.training.batch_size > 0
        assert self.reward.outcome_weight + self.reward.process_weight == 1
        assert self.search.num_simulations > 0
