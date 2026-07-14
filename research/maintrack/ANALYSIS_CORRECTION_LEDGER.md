# VERA Analysis Correction Ledger

## 2026-07-13: External violation versus unsupported deployment

The locked preregistration said an edit deployed into an external environment
absent from certification would count as unsafe. That is correct as a protocol
decision: no validation-only method can certify the edit without a structural
assumption. It is not, by itself, evidence that the edit violates the measured
external target-harm or leakage threshold.

Before running the aggregate real-study analysis, the analyzer was changed to
report both endpoints:

- `false_acceptance`: deployment of an edit that fails a measured external
  contract.
- `preregistered_unsafe_deployment`: deployment of an edit that either fails a
  measured external contract or enters an unsupported environment.

The paper must use the first endpoint for claims about observed contract
violations and the second only for claims about certifiability. The preregistered
endpoint remains in the receipts and report for auditability.

## 2026-07-13: Dependence across registered configurations

Threshold pairs and nested validation fractions reuse the same seed-level
fits and examples. Configuration-level Clopper--Pearson intervals and McNemar
tests therefore do not have independent Bernoulli trials. The analyzer retains
those preregistered calculations as diagnostics, but adds an exact sign-flip
test that treats each seed as one independent block. Inferential claims must be
governed by the seed-blocked result. The final confirmation uses eight untouched
blocks and a one-sided test fixed before those seeds run; a sign-only test is
reported as a secondary sensitivity.

## 2026-07-13: Exact external-outcome timing

The claim-grade runner fixes the preregistration, upstream commit, edit,
post-edit probes, and certification arrays before it computes external
outcomes for that run. The unattended matrix nevertheless computes and stores
those external outcomes run by run; it does not physically defer all external
label access until all 200 runs finish. No aggregate analysis or parameter
change is performed during the matrix, and the preregistration hash prevents a
changed configuration from being labeled claim-grade.

The paper may therefore say that external labels are unused by edit
construction and candidate selection, and that aggregate external analysis was
deferred until matrix completion. It must not say that external labels were
physically inaccessible throughout the entire 200-run execution.

## 2026-07-14: Hierarchical shift-class clarification

Proof review before aggregate real-outcome analysis found that target and
leakage use different contract-specific reference laws. In the final balanced
protocol, target harm is audited under each environment law $P_g$, while each
attacker's leakage is the equal mean of robust recalls under the two source
laws $P_s$. The theorem therefore applies to any joint deployment law whose
relevant environment and source conditionals satisfy their separately declared
density-ratio bounds. Environment and source marginal weights are unrestricted,
but the theorem does not assert that every arbitrary collection of conditional
laws is jointly realizable. This clarification changes no receipt, threshold,
candidate, selection rule, or computed radius.

## 2026-07-14: Maximum-class leakage is a degenerate estimand

The locked parent analysis defined leakage as the maximum attacker correctness
over represented source classes. Aggregate analysis produced zero VERA
deployments, and a post-primary intersection-union power diagnostic also
produced zero. Diagnosis then identified a structural problem: for a binary
source, an information-free constant attacker has class recalls `(1, 0)`, so
the maximum-class endpoint equals one even though balanced accuracy is chance.
The endpoint confounds classifier bias with representation leakage.

The failed parent and IUT outputs are retained. Before computing a corrected
endpoint, a new protocol was hashed and publicly tagged at SHA-256
`4e3392b4793cbcb5665feb3941e3402624654c48f146b18f82555cc3b356a9fd`.
It defines each attacker's leakage as the equal-weight average of its two
robust class recalls. A constant attacker then scores 0.5, and source-prior
shift cannot change the estimand. Existing seeds are explicitly exploratory;
new seeds are required for confirmatory claims.

Camelyon17's external center-2 slice contains only one registered binary source
class. External balanced leakage is therefore not estimable there and must be
reported as `NA`; unsupported deployment remains a separate procedural result.

## 2026-07-14: Independent-audit display-name canonicalization

The first post-aggregate raw-array audit reported 4,320 candidate-row
mismatches and 23 selection mismatches. Every mismatch involved the two R-LACE
candidates: official receipts preserve the upstream provenance label
`R-LACE`, while the locked method registry and aggregate tables use the display
label `RLACE`. The 4,320 count is exactly two candidates across all 2,160
registered configurations.

Before changing the audit, a direct diagnostic canonicalized only those labels.
All 25,920 numeric, eligibility, radius, external-metric, family-size, and
sample-size comparisons then matched, and all 10,800 rule selections replayed.
The independent auditor was corrected to construct candidate and method labels
from the locked method registry plus each receipt's registered strength, just
as the aggregate analyzer does. No data, threshold, confidence bound,
selection, endpoint, pass condition, aggregate artifact, or manuscript claim
was changed. The failed first audit remains preserved in Git history alongside
the corrected rerun.
