import argparse
import sys
import os

# Ensure the current directory is in the system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(
        description="Run training or testing for the taxi driver classification."
    )
    
    parser.add_argument("mode", choices=["train", "test"], help="Choose 'train' or 'test'")
    parser.add_argument("--test_dir", default="./test_data", help="Path to the folder containing test CSV files")

    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training process...")
        train_model()
    elif args.mode == "test":
        print("Starting testing process...")
        test_model(args.test_dir)
    else:
        print("Invalid mode. Choose 'train' or 'test'.")

if __name__ == '__main__':
    main()
