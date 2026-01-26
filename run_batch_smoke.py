import pandas as pd
import os
import sys

if __name__ == '__main__':
    # Task F: Run batch processing smoke test
    print("Running Task F: Batch processing smoke test...")
    sys.path.insert(0, os.path.abspath("leakage_agent/src"))

    # Create example directory and files if they don't exist
    example_dir = "transform/examples"
    if not os.path.exists(example_dir):
        os.makedirs(example_dir)

    with open(os.path.join(example_dir, "batch1.csv"), "w") as f:
        f.write("email,label\nbob@example.com,bird")
    with open(os.path.join(example_dir, "batch2.csv"), "w") as f:
        f.write("secret_key,label\nmy-batch-secret,fish")

    try:
        from leakage_agent.batch_processor import BatchProcessor
        p = BatchProcessor(verbose=False)
        results = p.process_directory(example_dir)
        print(results)
    except Exception as e:
        print(f"Batch processing smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
