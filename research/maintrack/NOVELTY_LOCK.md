# VERA Novelty Lock

## Defensible Claim

VERA is not another unconditional erasure method. Its novelty is the
frontier-aware edit-or-abstain contract: estimate a source-leakage versus
target-utility frontier, certify a safe set under simultaneous intervals, pick
the smallest source-reducing edit that preserves target risk, or return an
auditable abstention certificate.

## Not Just Prior Erasure

INLP, LEACE, R-LACE/RLACE, TaCo, SPLINCE, and MANCE motivate the baseline space, but
the VERA claim is different. Those methods define erasure or task-preserving
projection procedures. VERA defines a decision layer over candidate erasers: it
can accept, reject, or abstain based on target-preservation and source-leakage
certificates.

## Not Just Domain Generalization

Domain generalization methods optimize prediction under distribution shift.
VERA audits whether a representation edit is safe after source or environment
information is reduced. The paper may compare against domain generalization
baselines only with matched representations, splits, metrics, and source labels.

## Medical Scope

Camelyon17-WILDS is now the high-stakes hospital-shift benchmark row. It
supports representation-reliability evidence only. Alzheimer or other medical
data can be added only when site, scanner, cohort, or protocol labels make the
source-shift question explicit. None of these rows establish clinical safety or
deployment readiness.

## Locked Boundary

Allowed: VERA introduces a certified frontier and abstention protocol for
target-preserving source removal.

Forbidden: VERA is universally state of the art, clinically safe, independent
of source labels, or a reference implementation of every concept-erasure
baseline.
