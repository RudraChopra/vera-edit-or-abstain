# Literature Collision Refresh: July 17, 2026

This memo records a targeted collision sweep for the current VERA main-track
package. It is not a full systematic review and does not replace external
expert review. Its purpose is narrower: check whether the manuscript still
openly acknowledges the closest risk-control, selective-abstention,
covariate-shift, and concept-erasure neighbors after the controlled-shift
follow-up was inserted.

## Search Scope

Searches were run on July 17, 2026 across primary or near-primary scholarly
sources, using the following query families:

- `Learn Then Test concept erasure representation edits`
- `distribution-free risk control concept erasure`
- `support-aware certification representation edits deployment shift`
- `bounded density ratio conformal risk control distribution shift`
- `MANCE Manifold Aware Concept Erasure arXiv 2607.03973`
- `R-LACE LEACE TaCo concept erasure`
- `High Probability Risk Control Under Covariate Shift`
- `Weighted Conformal Risk Control Covariate Shift`
- `Selective Conformal Risk Control`
- `Joint Adaptive Prediction Sets risk utility 2026`

## Nearby Literature Families Checked

| Family | Representative sources | Current status |
| --- | --- | --- |
| Finite-family risk control | Learn Then Test, RCPS, CRC, Pareto Testing | Already cited and explicitly disclaimed as prior machinery. |
| Selective or joint risk control | Selective CRC; joint adaptive selective conformal risk control; group-conditional PAC routing | Added Selective CRC to the main, ICLR, and NeurIPS related-work text. Existing joint/group citations remain. |
| Covariate-shift risk control | Weighted conformal/risk-control work, high-probability risk control under covariate shift, robust validation | Added weighted CRC under covariate shift to the main and venue-variant related work. Existing HPRC/robust-validation/fine-grained robust conformal citations remain. |
| Concept erasure | INLP, R-LACE, LEACE, TaCo, SPLINCE, KRaM, Obliviator, perfect-erasure limits, MANCE | Current paper already cites the active close neighbors, including 2025/2026 concept-erasure work. |
| Support and impossibility | Covariate/domain-shift impossibility, support-aware evaluation limits | Current paper already cites domain-adaptation impossibility and support-evaluation boundaries. |

## Concrete Manuscript Changes

Two citations were added to `references_verified.bib` and threaded through the
main and venue-variant related-work sections:

- Zecchin et al., "Generalization and Informativeness of Weighted Conformal
  Risk Control Under Covariate Shift" (`arXiv:2501.11413`).
- Xu, Guo, and Wei, "Selective Conformal Risk Control" (`arXiv:2512.12844`).

These additions strengthen the claim boundary. They do not change the VERA
contribution statement: VERA is not claiming novelty for finite-family testing,
weighted covariate-shift risk control, selective acceptance, or abstention
alone. The claim remains the paired representation-edit deployment certificate:
a support-aware vector envelope of groupwise bounded reweighting budgets for
paired target harm and a retrained attacker portfolio, with unsupported cells
reported as an impossibility boundary.

## Outcome

No newly found source made the current contribution non-novel as stated. The
closest risk-control neighbors reinforce the need for the current framing:
VERA must be presented as an application-and-extension layer over known
distribution-free risk-control machinery, not as an invention of finite-family
testing or abstention.

Remaining work: a cold expert reviewer must still try to kill this claim,
especially against recent selective CRC, weighted CRC under covariate shift,
and concept-erasure papers. This memo is an internal search checkpoint, not a
human novelty sign-off.
