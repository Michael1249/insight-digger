# Feature Specification: Exploratory Factor Analysis (EFA) for Psychological Traits Discovery

**Feature Branch**: `001-efa-analysis`  
**Created**: November 24, 2025  
**Status**: Draft  
**Input**: User description: "Implement Exploratory Factor Analysis (EFA) for discovering hidden psychological dimensions in soc opros survey data. The analysis should identify latent factors like authoritarianism, openness, or other psychological traits by finding correlations between survey responses. Must handle dynamic table dimensions (currently 265 statements from 15 respondents) with proper statistical validation and interpretable factor loadings."

## Clarifications

### Session 2025-11-24

- Q: For the correlation matrix calculation in FR-001, which correlation method should be used given the ordinal nature of Likert scale data? → A: Pearson correlations (standard in most factor analysis software)
- Q: When KMO < 0.6 or Bartlett's test fails (p ≥ 0.05), what should the system response be? → A: Show warnings but allow user to proceed with analysis
- Q: For FR-010 factor score computation, which method should be used to calculate respondent scores on each factor? → A: Regression method (precise, handles unique variances)
- Q: For FR-014 missing data handling, which approach should be prioritized when responses have missing values? → A: Pairwise deletion for correlations, listwise for factor scores

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Factor Discovery (Priority: P1)

A researcher loads the soc opros survey data and performs exploratory factor analysis to discover the underlying psychological dimensions that explain response patterns across any number of statements (N variables) from any number of respondents (N observations).

**Why this priority**: Core functionality that delivers immediate scientific value. Enables discovery of latent psychological traits without requiring prior hypotheses about factor structure.

**Independent Test**: Can be fully tested by loading survey data, running EFA with default parameters, and producing a factor solution with interpretable loadings. Delivers standalone value for psychological research.

**Acceptance Scenarios**:

1. **Given** soc opros data with N statements and M respondents, **When** researcher runs basic EFA, **Then** system produces 3-8 meaningful factors with loadings matrix (where factors ≤ min(N/3, M-1))
2. **Given** factor solution is generated, **When** researcher examines factor loadings, **Then** statements with loadings >0.4 are clearly grouped by psychological themes
3. **Given** EFA completes successfully, **When** researcher reviews factor interpretation, **Then** each factor explains at least 5% of total variance

---

### User Story 2 - Statistical Validation and Diagnostics (Priority: P2)

A researcher validates the appropriateness of factor analysis for the dataset and ensures statistical reliability of the factor solution through comprehensive diagnostics.

**Why this priority**: Essential for scientific rigor. Prevents invalid interpretations and ensures the factor solution is statistically sound and replicable.

**Independent Test**: Can be tested independently by running statistical tests on the correlation matrix and factor solution, producing clear pass/fail validation metrics.

**Acceptance Scenarios**:

1. **Given** soc opros dataset, **When** researcher runs factorability tests, **Then** KMO measure >0.6 and Bartlett's test p<0.05 confirm suitability
2. **Given** factor solution is extracted, **When** researcher checks reliability, **Then** each factor shows Cronbach's alpha >0.7 for internal consistency
3. **Given** multiple factor extraction methods tested, **When** researcher compares solutions, **Then** system recommends optimal number of factors using eigenvalue and scree plot criteria

---

### User Story 3 - Advanced Factor Interpretation and Visualization (Priority: P3)

A researcher explores factor solutions through interactive visualizations, rotates factors for clearer interpretation, and exports results for publication or further analysis.

**Why this priority**: Enhances usability and scientific communication. Enables deeper exploration and professional presentation of results.

**Independent Test**: Can be tested by generating factor plots, applying rotations, and exporting formatted results. Delivers value for research dissemination.

**Acceptance Scenarios**:

1. **Given** initial factor solution, **When** researcher applies varimax rotation, **Then** rotated solution shows clearer simple structure with higher loadings
2. **Given** factor solution exists, **When** researcher generates factor plots, **Then** biplot shows statement clustering and respondent positioning in factor space
3. **Given** completed analysis, **When** researcher exports results, **Then** formatted output includes loadings table, variance explained, and factor scores

---

### Edge Cases

- What happens when correlation matrix is singular or has perfect multicollinearity?
- How does system handle missing response data across statements?
- What occurs when sample size is too small relative to number of variables (N<3p rule)?
- How does system respond when factorability tests fail (KMO<0.6, Bartlett's p≥0.05) but user wants to proceed?
- What happens when all eigenvalues are below 1.0 (Kaiser criterion)?
- How does system handle extremely unbalanced data (e.g., N_statements >> N_respondents or vice versa)?
- What occurs when data dimensions change dynamically (e.g., new statements added to survey)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST calculate correlation matrix from Likert scale responses using Pearson correlations (standard method for factor analysis and psychological research)
- **FR-002**: System MUST perform factorability assessment using KMO measure and Bartlett's sphericity test before extraction
- **FR-003**: System MUST extract factors using principal axis factoring method with eigenvalue >1.0 criterion as default
- **FR-004**: Users MUST be able to specify number of factors to extract between 1 and min(N_statements/3, N_respondents-1) where N adapts to actual data dimensions
- **FR-005**: System MUST apply varimax rotation by default with option for oblimin rotation for correlated factors
- **FR-006**: System MUST calculate factor loadings matrix with loadings >0.4 highlighted as practically significant
- **FR-007**: System MUST compute communalities for each variable showing proportion of variance explained
- **FR-008**: System MUST calculate total variance explained by each factor and cumulative variance
- **FR-009**: Users MUST be able to generate scree plot for visual determination of optimal factor number
- **FR-010**: System MUST compute factor scores for each respondent using regression method (precise estimation that handles unique variances)
- **FR-011**: System MUST provide Cronbach's alpha reliability coefficient for each extracted factor
- **FR-012**: System MUST validate minimum sample size adequacy (N_respondents ≥ max(50, 5*N_statements) or provide warnings for smaller samples)
- **FR-013**: Users MUST be able to export factor solution as formatted tables and visualizations
- **FR-014**: System MUST handle missing data using pairwise deletion for correlation matrix computation and listwise deletion for factor score calculation
- **FR-015**: System MUST provide factor interpretation guidelines based on loading patterns

### Key Entities

- **Factor Solution**: Represents complete EFA results including loadings matrix, variance explained, rotation method, and extraction criteria
- **Factor Loading**: Correlation between each statement and extracted factor, indicating strength of relationship (range: -1 to +1)  
- **Communality**: Proportion of variance in each statement explained by all extracted factors combined
- **Factor Score**: Individual respondent's position on each psychological dimension derived from their response pattern
- **Correlation Matrix**: Square matrix showing relationships between all statement pairs, foundation for factor extraction

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researchers can identify 3-8 interpretable psychological factors explaining ≥60% of total variance in survey responses
- **SC-002**: Factor analysis completes processing of N×M data matrix in reasonable time (≤30s for N≤500, M≤100 on standard hardware)
- **SC-003**: 90% of extracted factors demonstrate internal consistency with Cronbach's alpha ≥0.70
- **SC-004**: Factor solutions show clear simple structure with ≥70% of statements loading >0.4 on primary factor
- **SC-005**: System correctly identifies when data is unsuitable for factor analysis (KMO<0.5) and provides clear guidance
- **SC-006**: Researchers can replicate published EFA results using equivalent methods and rotation criteria
- **SC-007**: Factor interpretation achieves 80% agreement between independent researchers using loading patterns
- **SC-008**: Exported results meet APA formatting standards for statistical reporting in psychological research

## Dependencies and Assumptions

### Technical Dependencies
- Python scientific computing stack (NumPy, SciPy, pandas)
- Factor analysis libraries (scikit-learn, factor_analyzer, or statsmodels)
- Visualization libraries (matplotlib, seaborn, plotly) for scree plots and factor diagrams
- Statistical testing capabilities for KMO and Bartlett's test

### Data Assumptions
- Likert scale responses can be treated as interval data for correlation analysis
- Linear relationships exist between observed variables and latent factors
- Factors are normally distributed in the population
- Sample represents broader population of interest for psychological trait measurement

### Statistical Assumptions
- Sample size (N_respondents) provides sufficient data for exploratory analysis relative to number of variables (N_statements)
- Statement set contains enough redundancy to reveal underlying factor structure
- Response patterns reflect genuine psychological traits rather than random responding
- Missing data occurs at random and does not bias factor structure
- Rule of thumb: N_respondents ≥ 5*N_statements for reliable factor extraction

## Scope Boundaries

### Included in Scope
- Exploratory factor analysis using classical methods (principal axis factoring)
- Orthogonal (varimax) and oblique (oblimin) rotation options
- Statistical validation and goodness-of-fit assessment
- Factor interpretation support with loading significance thresholds
- Export capabilities for scientific reporting

### Excluded from Scope
- Confirmatory factor analysis (CFA) with pre-specified models
- Advanced methods like maximum likelihood or robust estimators  
- Missing data imputation beyond basic mean substitution
- Cross-validation or factor invariance testing across subgroups
- Integration with external psychological assessment frameworks
