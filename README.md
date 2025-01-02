# O1-Blueprints-with-SFT-Dataset

A reinforcement learning project focused on developing an advanced AI agent that learns reasoning and problem-solving capabilities through interaction with the OpenO1-SFT dataset. This project implements a custom environment and training pipeline to create an agent that can understand, reason about, and solve complex tasks.

## Key Features

- **Custom RL Environment**: Implements a specialized environment for processing and learning from the OpenO1-SFT dataset
- **Transformer-based Policy**: Utilizes a transformer architecture for encoding input sequences and generating actions
- **PPO Implementation**: Uses Proximal Policy Optimization for stable and efficient training
- **Batched Processing**: Supports efficient batch processing of trajectories during training
- **Flexible Reward System**: Implements a customizable reward model for evaluating agent performance

## Goals

1. **Enhanced Reasoning**: Train an agent that can develop sophisticated reasoning capabilities similar to advanced language models
2. **Efficient Learning**: Implement efficient training mechanisms using PPO and batch processing
3. **Scalable Architecture**: Design a scalable system that can handle large datasets and complex tasks
4. **Reproducible Results**: Provide a clear framework for reproducing and building upon the training results

## Requirements

- Python 3.10 (recommended)
- Virtual Environment

## Installation

1. Create a virtual environment with Python 3.10:
```bash
py -3.10 -m venv venv_310
```

2. Activate the virtual environment:
```bash
# On Windows
.\venv_310\Scripts\activate

# On Unix or MacOS
source venv_310/bin/activate
```

3. Install dependencies:
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers datasets
```

## Project Structure

```
o1-Blueprints/
├── src/
│   ├── data/           # Dataset handling
│   ├── environment/    # RL environment implementation
│   ├── learning/       # Learning algorithms (PPO)
│   ├── policy/         # Policy network implementation
│   ├── reward/         # Reward model
│   ├── training/       # Training loop and utilities
│   └── utils/          # Configuration and helper functions
├── train.py           # Main training script
└── README.md
```

## Usage

To train the model:
```bash
python train.py --data_path OpenO1-SFT.jsonl --output_dir outputs
```

## Dataset

This project uses the [OpenO1-SFT dataset](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) from Hugging Face for training.

## Model Architecture

- Policy Network: Transformer-based architecture
- Learning Algorithm: Proximal Policy Optimization (PPO)
- Environment: Custom reasoning environment for processing sequential data

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License
