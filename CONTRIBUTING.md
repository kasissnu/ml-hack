# Team 32 collaboration guide

## Ownership

### Person 1 - ML pipeline

Work primarily in:

- `src/data/`
- `src/features/`
- `src/models/`
- `src/decoder/`
- `experiments/`
- `artifacts/`
- `reports/eda/`

Responsibilities: data validation, candidate generation, feature engineering, five-model comparison, Bayesian optimisation, HMM/Viterbi, and model export.

### Person 2 - API and UI

Work primarily in:

- `src/api/`
- `src/ui/`
- `tests/integration/`
- `deployment/`

Responsibilities: FastAPI endpoints, artifact loading, evaluator UI, route visualisation, startup/deployment, and end-to-end integration.

## Branches

```text
main
person1/ml-pipeline
person2/api-ui
```

Do not edit the other person's owned folders without coordinating first. Both people may edit `README.md`, `contracts/`, and `tests/` by agreement.

## Integration rule

Person 2 must depend only on the public prediction interface in `contracts/prediction_contract.py`. Person 1 may change internal model code without breaking the UI as long as that interface remains compatible.

## Commit style

Use small commits such as:

```text
feat(ml): add candidate generation
feat(api): add route prediction endpoint
test(integration): validate route response schema
docs: update setup instructions
```
