# Tasks: Exploratory Factor Analysis (EFA) for Psychological Traits Discovery

**Input**: Design documents from `/specs/001-efa-analysis/`
**Prerequisites**: plan.md (✅), spec.md (✅), research.md (✅), data-model.md (✅), contracts/ (✅)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and EFA module dependencies

- [X] T001 Install factor_analyzer library dependency in requirements.txt
- [X] T002 [P] Install additional EFA dependencies (scipy, matplotlib, seaborn) in requirements.txt
- [X] T003 [P] Verify existing pandas, numpy compatibility for factor analysis
- [X] T004 [P] Create tests/test_efa_analyzer.py placeholder
- [X] T005 [P] Create tests/test_factor_validator.py placeholder  
- [X] T006 [P] Create tests/test_integration_efa.py placeholder

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core EFA infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Create EFAAnalyzer base class structure in src/efa_analyzer.py
- [X] T008 [P] Create FactorValidator base class structure in src/factor_validator.py
- [X] T009 [P] Create EFAVisualizer base class structure in src/visualization_utils.py
- [X] T010 Implement data validation framework in src/efa_analyzer.py
- [X] T011 [P] Implement statistical validation framework in src/factor_validator.py
- [X] T012 [P] Setup error handling and warnings system for EFA operations
- [X] T013 Verify integration with existing src/soc_opros_loader.py
- [X] T014 [P] Create shared EFA data types and result structures in src/efa_analyzer.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Factor Discovery (Priority: P1) 🎯 MVP

**Goal**: Researchers can load survey data and perform basic EFA to discover psychological dimensions with interpretable factor loadings

**Independent Test**: Load soc opros data, run EFA with default parameters, produce factor solution with 3-8 meaningful factors explaining ≥60% variance

### Implementation for User Story 1

- [X] T015 [P] [US1] Implement correlation matrix calculation with Pearson correlations in src/efa_analyzer.py
- [X] T016 [P] [US1] Implement principal axis factoring extraction method in src/efa_analyzer.py
- [X] T017 [US1] Implement eigenvalue >1.0 criterion for factor number determination in src/efa_analyzer.py
- [X] T018 [US1] Implement factor loadings matrix computation in src/efa_analyzer.py
- [X] T019 [US1] Implement communalities calculation for variance explained in src/efa_analyzer.py
- [X] T020 [US1] Implement oblimin rotation method (default) in src/efa_analyzer.py
- [X] T021 [US1] Add factor loading significance highlighting (>0.4) in src/efa_analyzer.py
- [X] T022 [US1] Implement regression method factor score computation in src/efa_analyzer.py
- [X] T023 [US1] Create basic EFA notebook in notebooks/02_efa_basic.ipynb
- [X] T024 [US1] Add basic factor interpretation output formatting in src/efa_analyzer.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Statistical Validation and Diagnostics (Priority: P2)

**Goal**: Researchers can validate the appropriateness of factor analysis and ensure statistical reliability through comprehensive diagnostics

**Independent Test**: Run statistical tests on correlation matrix and factor solution, producing clear pass/fail validation metrics with KMO >0.6 and Bartlett's test p<0.05

### Implementation for User Story 2

- [X] T025 [P] [US2] Implement KMO measure calculation (overall and individual) in src/factor_validator.py
- [X] T026 [P] [US2] Implement Bartlett's sphericity test in src/factor_validator.py
- [X] T027 [US2] Implement sample size adequacy checking (N ≥ max(50, 5*variables)) in src/factor_validator.py
- [X] T028 [US2] Implement Cronbach's alpha reliability calculation per factor in src/factor_validator.py
- [X] T029 [US2] Implement correlation matrix singularity checking in src/factor_validator.py
- [X] T030 [US2] Add factor extraction method comparison (PAF vs ML) in src/efa_analyzer.py
- [X] T031 [US2] Implement scree plot eigenvalue analysis in src/visualization_utils.py
- [X] T032 [US2] Add data adequacy warning system with configurable thresholds in src/factor_validator.py
- [X] T033 [US2] Implement missing data handling (pairwise for correlations, listwise for scores) in src/efa_analyzer.py
- [X] T034 [US2] Create validation workflow integration with basic EFA in notebooks/02_efa_basic.ipynb
- [X] T035 [US2] Add comprehensive validation reporting in src/factor_validator.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Advanced Factor Interpretation and Visualization (Priority: P3)

**Goal**: Researchers can explore factor solutions through interactive visualizations, apply rotations, and export publication-ready results

**Independent Test**: Generate factor plots, apply varimax rotation, export formatted results for publication or further analysis

### Implementation for User Story 3

- [X] T036 [P] [US3] Implement scree plot visualization in src/visualization_utils.py
- [X] T037 [P] [US3] Implement factor loadings heatmap visualization in src/visualization_utils.py  
- [X] T038 [P] [US3] Implement biplot showing statements and respondents in factor space in src/visualization_utils.py
- [X] T039 [US3] Implement varimax rotation option for clearer simple structure in src/efa_analyzer.py
- [X] T040 [US3] Implement rotation comparison tools (oblimin vs varimax) in src/efa_analyzer.py
- [ ] T041 [US3] Add factor interpretation guidelines based on loading patterns in src/efa_analyzer.py
- [ ] T042 [US3] Implement publication-ready table formatting for factor loadings in src/efa_analyzer.py
- [ ] T043 [US3] Add APA-style statistical reporting format in src/efa_analyzer.py
- [ ] T044 [US3] Implement export capabilities (CSV, PNG, PDF) in src/visualization_utils.py
- [ ] T045 [US3] Create advanced EFA notebook in notebooks/03_efa_advanced.ipynb
- [ ] T046 [US3] Create factor interpretation guide notebook in notebooks/04_efa_interpretation.ipynb
- [ ] T047 [US3] Add interactive visualization options for Jupyter environment in src/visualization_utils.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final integration

- [ ] T048 [P] Comprehensive unit tests for EFAAnalyzer class in tests/test_efa_analyzer.py
- [ ] T049 [P] Comprehensive unit tests for FactorValidator class in tests/test_factor_validator.py  
- [ ] T050 [P] End-to-end integration tests covering all user stories in tests/test_integration_efa.py
- [ ] T051 [P] Performance optimization for large datasets (N>300 variables) across all modules
- [ ] T052 [P] Error handling improvements and user-friendly error messages
- [ ] T053 [P] Documentation updates in docstrings following NumPy style
- [ ] T054 Code cleanup and refactoring for maintainability across all EFA modules
- [ ] T055 [P] Validate notebooks against quickstart.md tutorial examples
- [ ] T056 [P] Add edge case handling for singular matrices and perfect multicollinearity
- [ ] T057 [P] Memory usage optimization for Binder compatibility
- [ ] T058 Run full quickstart.md validation and fix any issues

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 but is independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Builds on US1/US2 but is independently testable

### Within Each User Story

- Core implementation before notebooks
- Basic methods before advanced features  
- Statistical calculations before visualization
- Single rotation before rotation comparison
- Error handling integrated throughout

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Within each story, tasks marked [P] can run in parallel (different files, no dependencies)

---

## Parallel Example: User Story 1

```bash
# Launch correlation matrix and extraction methods together:
Task: "Implement correlation matrix calculation with Pearson correlations in src/efa_analyzer.py"
Task: "Implement principal axis factoring extraction method in src/efa_analyzer.py"

# Launch loadings and communalities calculations together:
Task: "Implement factor loadings matrix computation in src/efa_analyzer.py"  
Task: "Implement communalities calculation for variance explained in src/efa_analyzer.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (install dependencies)
2. Complete Phase 2: Foundational (CRITICAL - EFA infrastructure)
3. Complete Phase 3: User Story 1 (Basic Factor Discovery)
4. **STOP and VALIDATE**: Test basic EFA on soc opros data
5. Demo psychological factor discovery capability

### Incremental Delivery

1. Complete Setup + Foundational → EFA infrastructure ready
2. Add User Story 1 → Test basic factor discovery → Demo (MVP!)
3. Add User Story 2 → Test statistical validation → Demo enhanced reliability
4. Add User Story 3 → Test visualization and export → Demo publication-ready analysis
5. Each story adds scientific value without breaking previous functionality

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Basic Factor Discovery)
   - Developer B: User Story 2 (Statistical Validation) 
   - Developer C: User Story 3 (Visualization & Interpretation)
3. Stories complete and integrate independently for comprehensive EFA capability

---

## Task Count Summary

- **Total Tasks**: 58
- **Setup Phase**: 6 tasks
- **Foundational Phase**: 8 tasks  
- **User Story 1**: 10 tasks
- **User Story 2**: 11 tasks
- **User Story 3**: 12 tasks
- **Polish Phase**: 11 tasks

**Parallel Opportunities**: 32 tasks marked [P] can run in parallel with others in their phase

**MVP Scope (Recommended)**: Phases 1-3 (24 tasks) delivers basic factor discovery functionality

**Independent Test Criteria**:
- US1: Load data → run EFA → produce interpretable factors
- US2: Validate data suitability → assess factor reliability → pass/fail metrics  
- US3: Generate visualizations → apply rotations → export publication results

---

## Notes

- [P] tasks = different files, no dependencies within phase
- [Story] label maps task to specific user story for traceability  
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Integration with existing `soc_opros_loader.py` maintains project structure
- All dependencies available via conda/pip for Binder compatibility