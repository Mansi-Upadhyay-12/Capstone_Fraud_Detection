"""
Root launcher for Stage 4H: Distribution-Robust Feature Learning & Hybrid Model Investigation
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train_stage4h import run_stage4h

if __name__ == "__main__":
    try:
        is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
        run_stage4h(smoke_test=is_smoke)
    except Exception as e:
        print(f"\n[ERROR] run_stage4h_experiments failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
