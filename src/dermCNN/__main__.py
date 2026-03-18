"""Command-line interface entry point for the DermCNN training pipeline.

This script allows users to start the training process directly from the 
terminal by executing the module. It provides arguments to select the 
classification mode (binary or malignant_only).
"""

import argparse
from .train import train

def main() -> None:
    """Parses command-line arguments and initiates the training process.
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Train the DermCNN cascade classification system."
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['binary', 'malignant_only'], 
        default='binary',
        help="Select the training mode: 'binary' (benign vs. malignant) or 'malignant_only' (multi-class skin cancer classification)."
    )
    
    args = parser.parse_args()
    train(mode=args.mode)

if __name__ == "__main__":
    main()