# Team 32 Phase 1 Findings

Generated from `scripts/run_phase1.py` using all 80 training trips and all 10 student-test trips.

## Dataset structure

Each trip contains:

- `.track`: longitude, latitude, timestamp in seconds
- `.nodes`: longitude, latitude indexed by row number
- `.arcs`: from-node ID and to-node ID indexed by row number
- `.route`: ordered reference arc IDs

All 90 available trips contain all four files. Every route arc ID and every arc endpoint was valid in the audit.

## Aggregate results

| Partition | Trips | GPS rows | Route rows | Mean weak components | Total route transition breaks | Mean route midpoint distance | Mean route coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 31,336 | 39.01 | 13 | 25.70 m | 91.12% |
| Student test | 10 | 19,160 | 3,477 | 22.70 | 1 | 27.89 m | 90.46% |

Additional observations:

- No missing GPS values were found.
- No invalid GPS coordinates were found.
- No isolated road-network nodes were found in the audited trips.
- GPS sampling is usually close to 1 second, but long gaps exist. The largest observed training gap is 153,064 seconds, so time gaps must be treated explicitly.
- The reference route contains 13 non-connected consecutive arc transitions in training and one in student test. The supplied route should therefore be treated as reference information with known limitations, consistent with the challenge guidance.

## Route-to-GPS alignment decision

The route length differs from the number of GPS observations, so `.route[i]` must not be treated as the label for `.track[i]`.

For Phase 2, use this label and candidate alignment procedure:

1. Build the geometry of each ordered reference route arc from its endpoint nodes.
2. For every GPS observation, find the closest reference route arc by geometric distance.
3. Enforce non-decreasing route order over time so GPS noise cannot move the assigned route position backwards.
4. Keep the original route order and allow repeated assignment of one arc to multiple GPS observations.
5. Track route transition breaks and large time gaps as quality flags rather than silently repairing the supplied reference route.

This is a training-label alignment heuristic, not a claim that the supplied route is independently verified ground truth. It will be evaluated against candidate miss rate and sequence continuity before model training proceeds.

## Generated artifacts

- `trip_audit.csv`: one validation and alignment record per trip
- `partition_summary.csv`: aggregate partition statistics
- `trip_00000001_track_route.html`: Folium map with GPS track and reference route

