# FARO Algorithm Spec

## Inputs

FARO receives frozen representations `z`, target labels `y`, source labels
`s`, locked train/validation/external splits, a candidate eraser family, and
metric tolerances for target preservation and leakage reduction.

## Candidate Frontier

For each candidate edit strength, FARO trains fresh target and source probes on
the edited training split and evaluates them on validation and external splits.
The frontier is the non-dominated set of edits that trade off target balanced
accuracy, worst target-source accuracy, and source leakage balanced accuracy.

## Selection Rule

FARO selects the strongest leakage-reducing frontier point whose target metrics
remain inside the pre-registered tolerance. If several candidates satisfy the
same tolerance, FARO chooses the one with the lowest validation source leakage
and reports external metrics only after the selection rule is fixed.

## Abstention Rule

FARO returns ABSTAIN when the frontier has no candidate edit whose confidence
interval satisfies both target-preservation and leakage-reduction constraints.
ABSTAIN is an algorithmic output, not a failed run.

## Output

The output is one of `EDIT` with the selected candidate and metrics, or
`ABSTAIN` with the failed constraints and calibration evidence.
