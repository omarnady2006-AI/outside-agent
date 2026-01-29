# Threat Model - Hybrid Data Governance Agent

## Overview

This document defines the threat model for the synthetic data governance system, including adversary assumptions, attack vectors, and security guarantees.

## System Scope

The governance agent evaluates synthetic datasets for:
- **Privacy leakage risk**: Near-duplicates, membership inference, attribute inference
- **Statistical fidelity**: Distribution preservation, correlation maintenance  
- **Semantic validity**: Business logic, domain constraints
- **Utility preservation**: ML task performance

## Adversary Model

### Adversary Capabilities

We assume an adversary who:

1. **Has access to**:
   - The synthetic dataset (full access)
   - Public information about the data generation process
   - General knowledge of the original data domain

2. **Does NOT have access to**:
   - The original raw dataset
   - Statistical profiles (stored securely)
   - Internal system logs and audit trails

3. **Can perform**:
   - Statistical analysis on synthetic data
   - Membership inference attacks
   - Attribute inference attacks
   - Record linkage attacks
   - Distribution comparison attacks

### Adversary Goals

The adversary aims to:
- **Re-identify individuals** from the original dataset
- **Infer sensitive attributes** not present in synthetic data
- **Determine membership** (was this record in the original data?)
- **Reconstruct original values** from synthetic transformations

## Attack Vectors

### 1. Membership Inference Attack

**Description**: Adversary trains a classifier to distinguish synthetic from original records.

**Risk Level**: HIGH

**Mitigations**:
- Membership inference AUC score ≤ 0.6 (close to random guessing 0.5)
- Monitor near-duplicate rates
- Enforce minimum distance to original records

**Detection**: Computed automatically in `PrivacyRiskMetrics`

---

### 2. Attribute Inference Attack

**Description**: Adversary uses correlations in synthetic data to infer missing sensitive attributes.

**Risk Level**: MEDIUM

**Mitigations**:
- Correlation preservation checks
- Cross-field logic validation
- Limit feature importance leakage

**Detection**: Attribute inference accuracy monitored in utility metrics

---

### 3. Near-Duplicate Re-identification

**Description**: Synthetic records are too similar to original records, enabling re-identification.

**Risk Level**: CRITICAL

**Mitigations**:
- Row hash comparison
- Nearest-neighbor distance thresholds (min distance ≥ 0.5 in standardized space)
- Near-duplicate rate must be < 1%

**Detection**: `detect_near_duplicates()` in `PrivacyRiskMetrics`

---

### 4. Distribution-Based Inference

**Description**: Adversary uses statistical distributions to narrow down possible original values.

**Risk Level**: MEDIUM

**Mitigations**:
- KL divergence monitoring
- Wasserstein distance checks
- Histogram overlap thresholds

**Detection**: Statistical fidelity metrics

---

### 5. Linkage Attack

**Description**: Adversary links synthetic records to external datasets using quasi-identifiers.

**Risk Level**: HIGH (depends on external data availability)

**Mitigations**:
- Tokenization of identifiers
- Bucketing of quasi-identifiers (age, location)
- Generalization schemes

**Detection**: Not automatically detected - requires domain-specific rules

---

## Security Guarantees

### What We Guarantee

1. **No Raw Data Storage**
   - Original datasets are NEVER persisted
   - Only statistical profiles (mean, variance, hashes) are stored
   - Verified by `SecurityConstraintTests`

2. **LLM Privacy**
   - LLM NEVER receives PII or raw data values
   - Only aggregate metrics sent to LLM
   - All prompts sanitized via `_sanitize_metrics()`

3. **Audit Trail Completeness**
   - All evaluations logged
   - All LLM interactions recorded
   - All threshold changes tracked

4. **Deterministic Privacy Checks**
   - Privacy computation is rule-based (no LLM bias)
   - Reproducible with same random seed
   - No non-deterministic privacy decisions

### What We Do NOT Guarantee

1. **Perfect Re-identification Prevention**
   - Cannot prevent all possible linkage attacks
   - External data availability affects risk

2. **Semantic Completeness**
   - Domain-specific rules must be manually configured
   - Cannot auto-detect all business logic

3. **Utility-Privacy Optimality**
   - Trade-offs are context-dependent
   - Recommendations are guidance, not guarantees

## Leakage Definition

### What Constitutes Leakage?

A synthetic dataset is considered "leaked" if:

1. **Exact matches**: Any synthetic record exactly matches an original record (hash collision)
2. **High similarity**: > 1% of synthetic records have Jaccard similarity > 0.9 to original records
3. **Membership inference**: Classifier can distinguish synthetic vs real with AUC > 0.7
4. **Semantic violations**: Domain constraints violated (indicates generation flaws)

### Risk Levels

- **Acceptable**: Privacy score ≥ 0.8, near-duplicates < 1%, membership AUC ≤ 0.6
- **Warning**: Privacy score 0.6-0.8, requires review
- **Critical**: Privacy score < 0.6, immediate regeneration required

## Protected Attributes

The following attribute types MUST be protected:

1. **Direct Identifiers**: Names, SSNs, email addresses → TOKENIZE or DROP
2. **Quasi-Identifiers**: Age, ZIP code, dates → BUCKET or GENERALIZE
3. **Sensitive Attributes**: Medical, financial, behavioral data → Explicit handling required

## Security Boundaries

### Trusted Components

- RuleEngine (deterministic, auditable)
- DataProfiler (no raw storage)
- Metric calculators (open-source algorithms)

### Untrusted Components

- GovernanceAgent (LLM provider)
  - Mitigation: Sanitize all inputs
  - Mitigation: Use local Ollama by default
  - Mitigation: Audit all LLM interactions

### Trust Assumptions

- Configuration files (YAML) are trusted
- Python dependencies are from PyPI (supply chain trust)
- Local filesystem is secure

## Compliance Considerations

This system supports but does not guarantee compliance with:

- **GDPR**: Article 25 (data minimization), Article 32 (pseudonymization)
- **CCPA**: De-identification requirements
- **HIPAA**: Safe Harbor method (if configured correctly)

**Note**: Compliance requires proper configuration and domain-specific validation.

## Incident Response

If leakage is detected:

1. **Immediate**: QUARANTINE synthetic dataset
2. **Within 1 hour**: Review audit logs
3. **Within 24 hours**: Root cause analysis
4. **Before re-generation**: Update policies and thresholds

## References

- [k-Anonymity](https://epic.org/privacy/reidentification/ohm_article.pdf)
- [Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Membership Inference Attacks](https://arxiv.org/abs/1610.05820)
- [Synthetic Data Best Practices](https://www.nist.gov/privacy-framework)
