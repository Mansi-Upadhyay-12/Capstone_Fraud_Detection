"""
Root launcher for Stage 4G: Temporal Distribution Shift & Post-Shutdown Investigation
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.diagnose_stage4g import run_stage4g_diagnostics

if __name__ == "__main__":
    try:
        run_stage4g_diagnostics()
    except Exception as e:
        print(f"\n[ERROR] run_stage4g_diagnostics failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
