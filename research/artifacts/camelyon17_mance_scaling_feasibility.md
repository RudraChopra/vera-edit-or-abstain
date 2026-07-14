# Camelyon17 MANCE++ Scaling Feasibility

Generated at UTC: `2026-07-14T00:53:00.440505+00:00`
Recommendation: `use the full no-cap Camelyon17 MANCE++ receipt as the current reference row`

| Train | Validation | External | Runtime | Work units |
| ---: | ---: | ---: | ---: | ---: |
| 20000 | 8000 | 8000 | 0.02 h | 3600000000 |
| 40000 | 16000 | 16000 | 0.05 h | 14400000000 |
| 80000 | 24000 | 24000 | 0.36 h | 51200000000 |

## Full No-Cap Projection

- Full train/validation/external counts: `{'train': 302436, 'validation': 68464, 'external': 85054}`
- Linear lower-bound estimate: `4.83 h`
- Recent superlinear estimate: `19.78 h`
- Recent scaling exponent: `1.542`
- Full no-cap completed: `True`
- Full no-cap observed runtime: `3.27 h`

## Boundary

The full no-cap MANCE++ Camelyon17 receipt is materialized and claim-grade under FARO's frozen-representation protocol.
