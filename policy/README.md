# Policy Configuration

## Purpose
The `policy/` directory manages the governance logic for the Auto-Supervisor Agent. Its primary role is to audit, sanitize, and gate candidate dataset copies before they are exposed to the outside environment. By enforcing strict transformation rules and safety checks, this module ensures that sensitive data is handled according to organizational standards, preventing leaks while maintaining dataset utility.

## Folder Structure
- **versions/**: Contains immutable snapshots of policy configurations (e.g., `v1/`, `v2/`).
- **current**: A pointer or symlink indicating the active policy version used by the agent.
- **rules.yaml**: (Within version folders) Maps specific data fields or patterns to their respective safety actions.
- **patterns.json**: (Within version folders) Stores regex and logic for identifying PII, PHI, and secrets.

## Supported Actions
- **TOKENIZE_DET**: Replaces sensitive data with deterministic tokens to maintain referential integrity.
- **DROP**: Completely removes the field from the dataset copy.
- **BUCKET**: Converts granular values (e.g., exact ages or timestamps) into broader ranges.
- **GENERALIZE**: Replaces specific attributes with higher-level categories to reduce uniqueness.
- **KEEP**: Permits the data to pass through without any modification.
- **QUARANTINE**: Blocks the entire record and flags it for manual security intervention.

## Versioning Rules
1. **Create**: Generate a new subdirectory under `versions/` using a sequential identifier (e.g., `v_next`).
2. **Modify**: All logic changes must occur within a new version; never modify an existing version folder.
3. **Switch**: Update the `current` file at the root of the `policy/` folder to point to the new version directory.

## Safety Rules
- **No Raw Logging**: Under no circumstances should raw sensitive values be written to logs or metadata.
- **Secret Enforcement**: Detection of credentials, private keys, or API secrets must always trigger a `QUARANTINE` action.
