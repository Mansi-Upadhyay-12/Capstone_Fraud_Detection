"""
Root launcher for Stage 5: Explainable AI & Fraud Ring Detection
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

for d in ["models", "reports", "results"]:
    os.makedirs(os.path.join(project_root, d), exist_ok=True)

from src.explainability_and_rings import run_stage5_explainability_and_rings

if __name__ == "__main__":
    try:
        run_stage5_explainability_and_rings()
    except Exception as e:
        print(f"\n[ERROR] explain_fraud_rings execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
