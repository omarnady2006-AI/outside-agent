import sys
import os
sys.path.insert(0, os.path.abspath("leakage_agent/src"))
from leakage_agent import LeakageValidator
v = LeakageValidator()
print(f"validator.policy_dir={v.policy_dir}")
if hasattr(v.pipeline, 'policy_dir'):
    print(f"pipeline.policy_dir={v.pipeline.policy_dir}")
else:
    print("pipeline.policy_dir=MISSING")
