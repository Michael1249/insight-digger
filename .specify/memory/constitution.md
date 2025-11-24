# Insight Digger Constitution

## Core Principles

### I. Requirements-First Development
Every notebook MUST start with documented data requirements and analysis objectives before any code is written. This includes expected data schema, source requirements, and clear success criteria. Notebooks that fail to document requirements upfront cannot be merged to main branch.

### II. Structure Preservation (NON-NEGOTIABLE)
The project structure (`/notebooks`, `/src`, `/data`, `/config`) MUST be preserved across all development. No reorganization or consolidation of core directories. Shared utilities belong in `/src`, data connectors in `/config`, raw/processed data separation maintained.

### III. Binder-First Compatibility
ALL notebooks MUST work in Binder cloud environment without local dependencies. This means no local file assumptions, proper error handling for missing credentials, and graceful degradation when external services are unavailable.

### IV. Data Source Resilience
Notebooks MUST include data validation and schema checking before processing. When primary data sources fail, notebooks MUST clearly indicate the failure and provide actionable error messages. Demo data fallbacks encouraged but not required.

### V. Iterative Development & Error Transparency
Code MUST fail fast and provide clear error messages to enable quick debugging. When data schema changes are detected (new columns, missing fields), notebooks MUST stop execution with specific information about what changed, allowing rapid iteration.

## Development Standards

All notebooks are independent but should leverage shared utilities from `/src` when appropriate. Self-contained approach preferred unless clear reusability benefits exist. Error handling and validation required in all data processing steps.

## Collaboration Guidelines

Project supports multiple contributors but assumes single primary maintainer. Contributors can experiment freely but finished work should create duplicates rather than modify existing stable notebooks. No formal review process required for experimentation, but production notebooks benefit from peer validation.

## Quality Assurance

Notebook templates encouraged but not mandated - flexibility preserved for independent analysis needs. Automated testing implemented for shared utilities in `/src`. Deployment pipeline includes basic smoke tests to verify notebooks can execute in Binder environment without critical failures.

## Governance

Constitution preserves project flexibility while ensuring reliability for public sharing via Binder. Structure decisions favor maintainability and ease of debugging. When in doubt, choose the option that makes iteration faster and errors more transparent.

**Version**: 1.0.0 | **Ratified**: 2025-11-24 | **Last Amended**: 2025-11-24
