# Paper A Lock

Paper A is the FARO method paper. Science-fair framing is out of scope.

## Locked Thesis

FARO converts representation erasure from "apply an edit and hope the target
task survives" into a certified selection problem. It evaluates a candidate
frontier of edits, accepts only those that preserve target performance under a
pre-specified tolerance while reducing source leakage, and returns ABSTAIN when
the frontier contains no certified safe edit.

## Contribution Sentence

FARO is an edit-or-abstain protocol for representation interventions that is
orthogonal to INLP, LEACE, RLACE, TaCo, SPLINCE, and MANCE++: those methods
construct candidate edits, while FARO certifies whether any candidate edit is
safe enough to deploy under the benchmark's target, source, and calibration
constraints.

## Evidence Lock

The locked evidence package includes official Waterbirds, official Camelyon17,
synthetic abstention certificates, real benchmark abstention stress tests,
claim-ledger audits, reproducibility audits, and the MANCE++ reference package
described in `MANCE_REFERENCE_STATUS.md`.
