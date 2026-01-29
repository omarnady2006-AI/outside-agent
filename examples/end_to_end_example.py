"""
Example: End-to-End Synthetic Data Governance Workflow

This script demonstrates the complete workflow of the hybrid governance agent:
1. Create statistical profile from original data
2. Evaluate synthetic data
3. Get LLM agent recommendations
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing governance_core
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np

from governance_core import (
    RuleEngine,
    GovernanceAgent,
    DataProfiler,
    AuditLogger
)

def generate_example_data():
    """Generate example datasets for demonstration."""
    np.random.seed(42)
    
    # Create original dataset
    n_original = 1000
    original_df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_original),
        'salary': np.random.normal(60000, 20000, n_original),
        'years_experience': np.random.randint(0, 40, n_original),
        'satisfaction_score': np.random.randint(1, 11, n_original),
        'label': np.random.choice(['A', 'B', 'C'], n_original)
    })
    
    # Create synthetic dataset (with some controlled leakage for demo)
    n_synthetic = 800
    synthetic_df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_synthetic),
        'salary': np.random.normal(62000, 18000, n_synthetic),  # Slightly different
        'years_experience': np.random.randint(0, 40, n_synthetic),
        'satisfaction_score': np.random.randint(1, 11, n_synthetic),
        'label': np.random.choice(['A', 'B', 'C'], n_synthetic)
    })
    
    # Add some near-duplicates (simulate leakage)
    # Instead of assignment, we'll replace specific rows
    num_duplicates = 5
    duplicate_indices = np.random.choice(len(original_df), num_duplicates, replace=False)
    
    for i, orig_idx in enumerate(duplicate_indices):
        synthetic_df.loc[i, 'age'] = original_df.loc[orig_idx, 'age']
        synthetic_df.loc[i, 'salary'] = original_df.loc[orig_idx, 'salary']
        synthetic_df.loc[i, 'years_experience'] = original_df.loc[orig_idx, 'years_experience']
        synthetic_df.loc[i, 'satisfaction_score'] = original_df.loc[orig_idx, 'satisfaction_score']
        synthetic_df.loc[i, 'label'] = original_df.loc[orig_idx, 'label']
    
    return original_df, synthetic_df


def main():
    print("=" * 80)
    print("HYBRID DATA GOVERNANCE AGENT - END-TO-END EXAMPLE")
    print("=" * 80)
    
    # Step 1: Generate example data
    print("\n📊 Generating example datasets...")
    original_df, synthetic_df = generate_example_data()
    print(f"   Original: {len(original_df)} rows")
    print(f"   Synthetic: {len(synthetic_df)} rows")
    
    # Step 2: Create data profile
    print("\n🔧 Creating statistical profile of original data...")
    profiler = DataProfiler(include_row_hashes=True)
    
    profile = profiler.create_profile(
        df=original_df,
        profile_id="example_original_001",
        include_value_hashes=False  # Not needed for this example
    )
    
    print(f"   Profile created: {profile.profile_id}")
    print(f"   Fields profiled: {', '.join(profile.column_names)}")
    
    # Optional: Save profile for later use
    profiles_dir = Path("example_outputs/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile.save(str(profiles_dir / "original_profile.json"))
    print(f"   Saved to: {profiles_dir / 'original_profile.json'}")
    
    # Step 3: Initialize audit logger
    print("\n📋 Initializing audit logger...")
    audit_logger = AuditLogger(output_dir="example_outputs/audit_logs")
    print(f"   Audit session: {audit_logger.session_file}")
    
    # Step 4: Run rule engine evaluation
    print("\n🔍 Running Rule Engine evaluation...")
    print("   This computes all metrics deterministically (no LLM)\n")
    
    engine = RuleEngine(config=None, audit_logger=audit_logger)
    
    result = engine.evaluate_synthetic_data(
        synthetic_df=synthetic_df,
        original_df=original_df,  # Could also use: original_profile=profile
        eval_id="example_eval_001",
        target_column="label"  # For utility metrics
    )
    
    print("\n" + "=" * 80)
    print("RULE ENGINE RESULTS")
    print("=" * 80)
    
    print(f"\n🔒 Privacy Score: {result['privacy_score']:.3f}")
    print(f"   Risk Level: {result['leakage_risk_level'].upper()}")
    print(f"   Near-duplicates: {result['privacy_risk']['near_duplicates_count']} "
          f"({result['privacy_risk']['near_duplicates_rate']:.2%})")
    
    if result.get('privacy_risk', {}).get('membership_inference_auc'):
        print(f"   Membership Inference AUC: "
              f"{result['privacy_risk']['membership_inference_auc']:.3f}")
    
    print(f"\n📈 Utility Score: {result.get('utility_score', 'N/A')}")
    if result.get('utility_score'):
        print(f"   Assessment: {result['utility_assessment'].upper()}")
        print(f"   Synthetic-trained accuracy: "
              f"{result['utility_preservation']['synthetic_train_real_test_accuracy']:.3f}")
        print(f"   Real-trained accuracy: "
              f"{result['utility_preservation']['real_train_real_test_accuracy']:.3f}")
    
    print(f"\n📊 Statistical Drift: {result['statistical_drift'].upper()}")
    
    print(f"\n⚠️  Semantic Violations: {result['semantic_violations']}")
    
    # Step 5: Run LLM agent interpretation (optional)
    print("\n" + "=" * 80)
    print("LLM AGENT INTERPRETATION")
    print("=" * 80)
    print("\n⚙️  Initializing Governance Agent (Ollama)...")
    print("   NOTE: This requires Ollama to be running locally")
    print("   If unavailable, fallback to rule-based interpretation\n")
    
    try:
        agent = GovernanceAgent(
            provider_type="ollama",
            audit_logger=audit_logger
        )
        
        print(f"✅ Agent initialized: {agent.provider.provider_name}\n")
        
        # Get interpretation
        interpretation = agent.interpret_metrics(
            metrics=result,
            context={
                "use_case": "ML training data",
                "sensitivity": "medium",
                "domain": "synthetic example"
            },
            eval_id=result['eval_id']
        )
        
        # Display structured decision
        import textwrap
        
        print(f"🎯 DECISION: {interpretation['decision']}")
        
        print(f"\n📋 Justification:")
        wrapped = textwrap.fill(interpretation['justification'], width=76, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)
        
        print(f"\n⚠️  Risk Assessment:")
        wrapped = textwrap.fill(interpretation['risk_assessment'], width=76, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)
        
        print(f"\n📊 Monitoring Recommendation:")
        wrapped = textwrap.fill(interpretation['monitoring_recommendation'], width=76, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)
        
    except Exception as e:
        print(f"⚠️  LLM Agent unavailable: {e}")
        print("   Continuing with rule-based evaluation only")
    
    # Step 6: Save full results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    import json
    output_dir = Path("example_outputs/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"evaluation_{result['eval_id']}.json"
    with open(result_file, 'w') as f:
        json_result = json.loads(json.dumps(result, default=str))
        json.dump(json_result, f, indent=2)
    
    print(f"\n💾 Evaluation result: {result_file}")
    print(f"📋 Audit log: {audit_logger.session_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if result['leakage_risk_level'] == 'critical':
        print("\n❌ RESULT: CRITICAL - Privacy risk too high")
        print("   Action: Regenerate synthetic data with stronger privacy controls")
    elif result['leakage_risk_level'] == 'warning':
        print("\n⚠️  RESULT: WARNING - Review recommended")
        print("   Action: Consider adjusting generation parameters")
    else:
        print("\n✅ RESULT: ACCEPTABLE - Data meets privacy requirements")
        print("   Action: Can proceed with synthetic data usage")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
