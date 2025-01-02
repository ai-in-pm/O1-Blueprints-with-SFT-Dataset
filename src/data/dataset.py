import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizer

class ReasoningDataset(Dataset):
    """Dataset for reasoning tasks from OpenO1-SFT format."""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess data from JSONL file."""
        examples = []
        try:
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    try:
                        example = json.loads(line.strip())
                        
                        # Extract instruction and output
                        instruction = example.get('instruction', '')
                        output = example.get('output', '')
                        
                        if instruction and output:
                            examples.append({
                                'input': instruction,
                                'output': output,
                                'metadata': {
                                    'id': str(i),
                                    'source': 'OpenO1-SFT'
                                }
                            })
                        
                        # Print progress
                        if len(examples) % 1000 == 0:
                            print(f"Loaded {len(examples)} examples")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing example: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error loading data file: {e}")
            raise
            
        print(f"Total examples loaded: {len(examples)}")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get preprocessed example."""
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize output
        output_encoding = self.tokenizer(
            example['output'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': output_encoding['input_ids'].squeeze(),
            'metadata': example['metadata']
        }

class PreferenceDataset(Dataset):
    """Dataset for preference learning."""
    
    def __init__(self,
                 preferred_trajectories: List[Dict],
                 non_preferred_trajectories: List[Dict]):
        assert len(preferred_trajectories) == len(non_preferred_trajectories)
        self.preferred = preferred_trajectories
        self.non_preferred = non_preferred_trajectories
        
    def __len__(self) -> int:
        return len(self.preferred)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get preference pair."""
        return {
            'preferred': self.preferred[idx],
            'non_preferred': self.non_preferred[idx]
        }

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader with default settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
