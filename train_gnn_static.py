"""
Root launcher for Stage 4A: Static Graph Neural Networks (GCN & GAT)
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

for d in ["models", "reports", "results"]:
    os.makedirs(os.path.join(project_root, d), exist_ok=True)

from src.train_gnn_static import run_stage4a

if __name__ == "__main__":
    try:
        is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
        run_stage4a(smoke_test=is_smoke)
    except Exception as e:
        print(f"\n[ERROR] train_gnn_static execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
