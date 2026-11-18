# Team 32 Phase 3 Feature Schema

Implemented in `src/features/build_features.py`.

The model input contains 13 features. Reference route IDs remain separate targets and are never included as model inputs.

| Feature | Meaning |
|---|---|
| `point_segment_distance_m` | Exact projected GPS point to candidate segment distance |
| `midpoint_distance_m` | GPS point to candidate segment midpoint distance |
| `along_segment_fraction` | Projected position between segment endpoints |
| `segment_length_m` | Candidate arc length |
| `gps_speed_mps` | GPS displacement divided by timestamp gap |
| `time_gap_s` | Time since the previous GPS observation |
| `gps_bearing_deg` | Bearing of the GPS displacement |
| `segment_bearing_deg` | Bearing from candidate segment start to end |
| `heading_difference_deg` | Absolute wrapped GPS-to-segment heading difference |
| `from_node_degree` | Road-network degree of the candidate start node |
| `to_node_degree` | Road-network degree of the candidate end node |
| `candidate_rank` | Rank from the candidate generator |
| `previous_candidate_connected` | Whether the candidate touches any endpoint in the previous observation's candidate set |

The first observation has zero displacement and zero time gap by definition. Large timestamp gaps are preserved as features and should not be silently filled or discarded.
