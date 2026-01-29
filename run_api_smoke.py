import pandas as pd
import os
import sys

# Task D: Run API smoke test
print("Running Task D: API smoke test...")
sys.path.insert(0, os.path.abspath("leakage_agent/src"))

try:
    from leakage_agent import LeakageValidator
    v = LeakageValidator()  # should auto-resolve policy_dir
    df = pd.DataFrame({"email":["alice@example.com"],"label":["cat"],"amount":[100]})
    res = v.validate(df, copy_id="smoke_api_001")
    print("decision:", res.decision)
    print("reason_codes:", getattr(res, "reason_codes", None))
    print("metrics keys:", list(getattr(res, "metrics", {}).keys())[:10])
except Exception as e:
    print(f"API Smoke test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
