"""
Root launcher for Stage 4F: Graph Model Performance Optimization
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train_stage4f import run_stage4f

if __name__ == "__main__":
    try:
        is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
        run_stage4f(smoke_test=is_smoke)
    except Exception as e:
        print(f"\n[ERROR] run_stage4f_optimization execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
