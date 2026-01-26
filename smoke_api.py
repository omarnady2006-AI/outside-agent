import pandas as pd
from leakage_agent import LeakageValidator
v = LeakageValidator()  # should auto-resolve policy_dir
df = pd.DataFrame({"email":["alice@example.com"],"label":["cat"],"amount":[100]})
res = v.validate(df, copy_id="smoke_api_001")
print("decision:", res.decision)
print("reason_codes:", getattr(res, "reason_codes", None))
print("metrics keys:", list(getattr(res, "metrics", {}).keys())[:10])
