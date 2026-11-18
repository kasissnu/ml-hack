# Team 32 Phase 2 Candidate Findings

The baseline generator is implemented in `src/features/candidates.py` and audited by `scripts/run_phase2.py`.

## Baseline method

- Project each trip into local metric coordinates.
- Query the nearest road-arc midpoints with a KD-tree.
- Compute exact point-to-segment distance, along-segment fraction, segment bearing, GPS bearing, and heading difference for each candidate.
- Use `K` nearest candidates per observation.
- Compare candidates with the provisional monotonic route alignment from Phase 1.

## Candidate coverage before alignment improvement

| K | Mean training miss rate | Mean student-test miss rate |
|---:|---:|---:|
| 8 | 22.84% | 20.34% |
| 12 | 20.51% | 17.98% |
| 16 | 19.06% | 16.94% |
| 24 | 17.13% | 15.53% |

The detailed results are in `candidate_summary_k8.csv`, `candidate_summary_k12.csv`, `candidate_summary_k16.csv`, and `candidate_summary_k24.csv`.

## Current decision

## Alignment improvement

The first alignment used one nearest route midpoint and a cumulative maximum. On looped or repeated geometry, one ambiguous point could force all later labels forward. The implementation now queries multiple nearby route positions and selects a fast monotonic sequence with a progress penalty.

With the improved alignment and `K=24` candidates:

| Partition | Mean miss rate | Maximum trip miss rate |
|---|---:|---:|
| Training | 5.39% | 56.35% |
| Student test | 4.64% | 15.80% |

`K=24` is the Phase 2 baseline. The remaining high-miss training trip is retained for analysis and is not silently removed. Models must report candidate miss rate because a reference arc absent from the candidate set cannot be recovered by the classifier.

Several trips have low geometric distance to the road network but high reference-arc miss rates. This indicates that the provisional nearest route-midpoint plus non-decreasing index rule can collapse onto an incorrect route position on loops or repeated geometry. The next task is to improve route-to-GPS alignment and compare candidate generation using nearby node incident arcs before beginning model training.
