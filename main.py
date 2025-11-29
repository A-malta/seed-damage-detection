import argparse
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.app.gui import run_gui
from src.config import MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Seed Damage Detection System")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'gui'], help="Mode to run the application")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'evaluate':
        evaluate_model(MODEL_PATH)
    elif args.mode == 'gui':
        run_gui()

if __name__ == "__main__":
    main()
