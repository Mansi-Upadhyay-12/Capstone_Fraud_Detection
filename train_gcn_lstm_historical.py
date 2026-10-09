import os
import sys

# Ensure src is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_gcn_lstm_historical import run_stage4b1_ablation

if __name__ == "__main__":
    run_stage4b1_ablation()
