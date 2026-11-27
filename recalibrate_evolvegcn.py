"""
Root launcher for Stage 4C Threshold Recalibration (Validation-Only 0.01-0.99)
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.recalibrate_evolvegcn import run_recalibration

if __name__ == "__main__":
    try:
        run_recalibration()
    except Exception as e:
        print(f"\n[ERROR] recalibrate_evolvegcn execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
