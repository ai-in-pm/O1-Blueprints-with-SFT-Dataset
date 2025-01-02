import gradio as gr
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json

from ..policy.base_policy import ReasoningPolicy
from ..utils.config import Config

class O1AgentUI:
    """User interface for interacting with the O1-like agent."""
    
    def __init__(self, model_path: str):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.policy = self._load_model(model_path)
        self.policy.eval()
        
        # Setup interface
        self.interface = self._create_interface()
    
    def _load_model(self, model_path: str) -> ReasoningPolicy:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        policy = ReasoningPolicy(checkpoint['config'])
        policy.load_state_dict(checkpoint['policy_state_dict'])
        return policy.to(self.device)
    
    def _create_interface(self) -> gr.Interface:
        """Create Gradio interface."""
        return gr.Interface(
            fn=self.generate_response,
            inputs=[
                gr.Textbox(
                    lines=5,
                    label="Task Description",
                    placeholder="Enter your task or question here..."
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    label="Temperature"
                ),
                gr.Checkbox(
                    label="Show intermediate steps",
                    value=True
                )
            ],
            outputs=[
                gr.Textbox(
                    lines=10,
                    label="Response"
                ),
                gr.JSON(
                    label="Reasoning Steps",
                    visible=True
                )
            ],
            title="O1-like Reasoning Agent",
            description="""
            This interface allows you to interact with an AI agent trained to replicate
            O1-like reasoning capabilities. The agent can handle complex tasks by breaking
            them down into steps and providing detailed explanations.
            """,
            examples=[
                ["Explain how a bicycle works.", 0.7, True],
                ["Solve this math problem: If x + 2 = 5, what is x?", 0.5, True],
                ["Write a short story about a robot learning to paint.", 0.8, True]
            ],
            theme="default"
        )
    
    def generate_response(self,
                         task: str,
                         temperature: float,
                         show_steps: bool) -> Tuple[str, Dict]:
        """Generate response for user input."""
        try:
            # Tokenize input
            inputs = self.policy.tokenizer(
                task,
                return_tensors="pt",
                max_length=self.config.model.max_seq_length,
                truncation=True
            ).to(self.device)
            
            # Generate response with intermediate steps
            with torch.no_grad():
                outputs = self.policy.generate(
                    inputs["input_ids"],
                    max_length=self.config.model.max_seq_length,
                    temperature=temperature,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode response
            response = self.policy.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )
            
            # Extract reasoning steps if requested
            steps = {}
            if show_steps:
                steps = self._extract_reasoning_steps(outputs)
            
            return response, steps
            
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e)}
    
    def _extract_reasoning_steps(self, outputs) -> Dict:
        """Extract intermediate reasoning steps from model outputs."""
        steps = {}
        
        # Extract attention weights or other relevant information
        # This is a placeholder - implement based on your model's output structure
        steps["attention"] = outputs.scores[0].mean(dim=0).tolist()
        
        return steps
    
    def launch(self, share: bool = False):
        """Launch the interface."""
        self.interface.launch(share=share)

def main():
    """Main entry point for the UI."""
    import argparse
    parser = argparse.ArgumentParser(description="O1-like Agent UI")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable link"
    )
    
    args = parser.parse_args()
    
    ui = O1AgentUI(args.model_path)
    ui.launch(share=args.share)

if __name__ == "__main__":
    main()
