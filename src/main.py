# src/main.py
import sys
from config import DATA_PATH, TEST_DATA_PATH, NUS, GAMMAS, MODEL_DIR, CONFIDENCE_THRESHOLD
from train_model import train_and_save_models_with_split
from test_model import bulk_test_with_confidence

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'

    print(f"Mode: {mode}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"TEST_DATA_PATH: {TEST_DATA_PATH}")
    print(f"MODEL_DIR: {MODEL_DIR}")

    if mode == 'train':
        print("Starting training...")
        train_and_save_models_with_split(DATA_PATH, NUS, GAMMAS, MODEL_DIR)
        print("Training completed!")
    elif mode == 'test':
        print("Starting testing...")
        bulk_test_with_confidence(TEST_DATA_PATH, NUS, GAMMAS, MODEL_DIR, CONFIDENCE_THRESHOLD)
        print("Testing completed!")
    else:
        print("Invalid mode. Use 'train' or 'test'.")
