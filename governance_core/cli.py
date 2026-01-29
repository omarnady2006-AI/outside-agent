"""
CLI for Hybrid Data Governance Agent

Provides command-line interface for:
- Evaluating synthetic datasets
- Creating data profiles
- Generating governance reports
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import logging

from governance_core import RuleEngine, GovernanceAgent, DataProfiler, AuditLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_evaluate(args):
    """Evaluate synthetic data against original or profile."""
    
    print("=" * 70)
    print("HYBRID DATA GOVERNANCE AGENT - SYNTHETIC DATA EVALUATION")
    print("=" * 70)
    
    # Load synthetic data
    print(f"\n📊 Loading synthetic data: {args.synthetic}")
    synthetic_df = pd.read_csv(args.synthetic)
    print(f"   Loaded {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns")
    
    # Load original data or profile
    original_df = None
    original_profile = None
    
    if args.original:
        print(f"\n📊 Loading original data: {args.original}")
        original_df = pd.read_csv(args.original)
        print(f"   Loaded {len(original_df)} rows, {len(original_df.columns)} columns")
    elif args.profile:
        print(f"\n📋 Loading original data profile: {args.profile}")
        from governance_core.data_profiles import DatasetProfile
        original_profile = DatasetProfile.load(args.profile)
        print(f"   Profile ID: {original_profile.profile_id}")
        print(f"   Original had {original_profile.row_count} rows")
    else:
        print("\n⚠️  WARNING: No original data or profile provided")
        print("   Privacy and utility metrics will be limited")
    
    # Initialize components
    audit_logger = AuditLogger(output_dir=args.audit_dir)
    
    # Initialize rule engine
    print(f"\n🔧 Initializing Rule Engine...")
    config = None  # Could load from args.policy_dir if needed
    engine = RuleEngine(config=config, audit_logger=audit_logger)
    
    # Run evaluation
    print(f"\n🔍 Running comprehensive evaluation...")
    print("   This may take a few minutes for large datasets...\n")
    
    result = engine.evaluate_synthetic_data(
        synthetic_df=synthetic_df,
        original_df=original_df,
        original_profile=original_profile,
        eval_id=args.eval_id,
        target_column=args.target_column
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n🔒 Privacy Score: {result['privacy_score']:.3f} (0.0 = critical, 1.0 = perfect)")
    print(f"   Risk Level: {result['leakage_risk_level'].upper()}")
    
    if result.get('utility_score'):
        print(f"\n📈 Utility Score: {result['utility_score']:.3f} (0.0 = no utility, 1.0 = perfect)")
        print(f"   Assessment: {result['utility_assessment'].upper()}")
    else:
        print(f"\n📈 Utility Score: N/A (original data required)")
    
    print(f"\n📊 Statistical Drift: {result['statistical_drift'].upper()}")
    print(f"\n⚠️  Semantic Violations: {result['semantic_violations']}")
    
    # LLM interpretation (if enabled)
    if args.enable_agent:
        print(f"\n🤖 Running LLM Agent interpretation...")
        try:
            agent = GovernanceAgent(
                provider_type=args.provider,
                audit_logger=audit_logger
            )
            
            interpretation = agent.interpret_metrics(
                metrics=result,
                context={"use_case": args.use_case, "sensitivity": args.sensitivity},
                eval_id=result['eval_id']
            )
            
            print("\n" + "=" * 70)
            print("LLM AGENT INTERPRETATION")
            print("=" * 70)
            print(f"\n{interpretation['overall_assessment']}")
            print(f"\n🎯 Recommendation: {interpretation['recommendation']}")
            
            if interpretation.get('key_issues'):
                print(f"\n⚠️  Key Issues:")
                for issue in interpretation['key_issues']:
                    print(f"   - {issue}")
            
            if interpretation.get('corrective_actions'):
                print(f"\n💡 Suggested Actions:")
                for action in interpretation['corrective_actions']:
                    print(f"   - {action}")
            
            print(f"\n📝 Explanation:")
            print(f"   {interpretation['explanation']}")
            
        except Exception as e:
            print(f"\n⚠️  LLM Agent unavailable: {e}")
            print(f"   Continuing with rule-based evaluation only")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"evaluation_{result['eval_id']}.json"
    with open(result_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_result = json.loads(json.dumps(result, default=str))
        json.dump(json_result, f, indent=2)
    
    print(f"\n💾 Results saved to: {result_file}")
    print(f"📋 Audit log: {audit_logger.session_file}")
    
    # Exit code based on risk level
    if result['leakage_risk_level'] == 'critical':
        print(f"\n❌ CRITICAL: Privacy risk too high")
        return 3
    elif result['leakage_risk_level'] == 'warning':
        print(f"\n⚠️  WARNING: Review recommended")
        return 2
    else:
        print(f"\n✅ ACCEPTABLE: Data meets privacy requirements")
        return 0


def cmd_create_profile(args):
    """Create statistical profile from original dataset."""
    
    print("=" * 70)
    print("DATA PROFILE CREATION")
    print("=" * 70)
    
    print(f"\n📊 Loading original data: {args.original}")
    original_df = pd.read_csv(args.original)
    print(f"   Loaded {len(original_df)} rows, {len(original_df.columns)} columns")
    
    print(f"\n🔧 Creating statistical profile...")
    print(f"   Include value hashes: {args.include_value_hashes}")
    print(f"   Include row hashes: {not args.no_row_hashes}")
    
    profiler = DataProfiler(include_row_hashes=not args.no_row_hashes)
    
    profile = profiler.create_profile(
        df=original_df,
        profile_id=args.profile_id,
        include_value_hashes=args.include_value_hashes
    )
    
    # Save profile
    output_path = Path(args.output)
    profile.save(str(output_path))
    
    print(f"\n✅ Profile created successfully")
    print(f"   Profile ID: {profile.profile_id}")
    print(f"   Rows: {profile.row_count}")
    print(f"   Columns: {profile.column_count}")
    print(f"\n💾 Saved to: {output_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Data Governance Agent for Synthetic Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate synthetic data against original
  python -m governance_core.cli evaluate --synthetic syn.csv --original orig.csv
  
  # Evaluate using pre-computed profile
  python -m governance_core.cli evaluate --synthetic syn.csv --profile orig_profile.json
  
  # Enable LLM agent interpretation
  python -m governance_core.cli evaluate --synthetic syn.csv --original orig.csv --enable-agent
  
  # Create data profile
  python -m governance_core.cli create-profile --original data.csv --output profile.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate synthetic data')
    eval_parser.add_argument('--synthetic', required=True, help='Path to synthetic CSV file')
    eval_parser.add_argument('--original', help='Path to original CSV file')
    eval_parser.add_argument('--profile', help='Path to original data profile JSON')
    eval_parser.add_argument('--target-column', help='Target column for utility metrics (default: label)')
    eval_parser.add_argument('--eval-id', help='Evaluation ID (default: auto-generated)')
    eval_parser.add_argument('--output', default='governance_outputs', help='Output directory')
    eval_parser.add_argument('--audit-dir', default='audit_logs', help='Audit log directory')
    eval_parser.add_argument('--enable-agent', action='store_true', help='Enable LLM agent interpretation')
    eval_parser.add_argument('--provider', default='ollama', choices=['ollama', 'anthropic', 'openai'], 
                             help='LLM provider (default: ollama)')
    eval_parser.add_argument('--use-case', default='general', help='Use case context')
    eval_parser.add_argument('--sensitivity', default='medium', choices=['low', 'medium', 'high'],
                             help='Data sensitivity level')
    
    # Create profile command
    profile_parser = subparsers.add_parser('create-profile', help='Create data profile')
    profile_parser.add_argument('--original', required=True, help='Path to original CSV file')
    profile_parser.add_argument('--output', required=True, help='Output profile JSON path')
    profile_parser.add_argument('--profile-id', required=True, help='Unique profile identifier')
    profile_parser.add_argument('--include-value-hashes', action='store_true',
                                help='Include value hashes (increases file size)')
    profile_parser.add_argument('--no-row-hashes', action='store_true',
                                help='Skip row hashes (disables near-duplicate detection)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'evaluate':
            return cmd_evaluate(args)
        elif args.command == 'create-profile':
            return cmd_create_profile(args)
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
