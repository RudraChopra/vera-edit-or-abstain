# MOSAIC: Data-Certified Stochastic Release under Structured Deployment Shift

MOSAIC (Minimax-Optimized Source-Agnostic Invariant Channels) is a
software-only method for deciding whether an edited representation is safe to
expose through a task-specific interface. It jointly selects a stochastic
finite-token channel and a task decoder, certifies them against the strongest
downstream source attacker, and returns `ABSTAIN` when source leakage or
worst-stratum utility cannot be certified under a registered deployment-shift
model.

This branch contains the AAAI 2027 research candidate, its theorem and proof
record, hash-locked protocols, official-method real-feature confirmation,
independent replay programs, receipts, figures, and named/anonymous manuscript
sources. It does not claim that conference acceptance or real-world safety is
guaranteed.

## What Is New

- One multinomial confidence event over fine-token laws covers every stochastic
  release channel selected from the same table, including a continuum of
  channels, without a channel-count correction.
- A labeled target-bridge program certifies the largest mass explained by one
  common pre-release transform. An exact envelope charges the remaining
  source-specific contamination and matches it with a task-utility certificate.
- A finite-alphabet optimizer globally selects the channel and decoder by linear
  optimization and abstains on unsupported, infeasible, or numerically
  inconsistent cases.
- No-free-lunch and missing-source theorems state when nontrivial certification
  is impossible.

The paper does **not** claim that stochastic mappings, Learn Then Test,
contraction coefficients, concept erasure, or abstention are individually new.
The narrow contribution is their adaptive, universal-attacker, shift-aware
combination and the transform-exact certificate.

## Headline Evidence

- Naive continuum selection falsely accepts 42.1% of 1,000 locked hard-boundary
  tables; MOSAIC records zero false acceptances.
- On 1,000 paired low-data tables, the transform-exact certificate safely
  deploys on 53.0%, versus 0.3% for the capacity-transfer fallback. At the next
  sample size, the rates are 99.9% and 57.8%.
- The transform-exact audit independently replays 10,000 decisions, 20,000
  source-leakage certificates, 40,000 utility certificates, and 20,000 external risks
  with zero mismatch.
- In a registered 100-job real-data bridge study, unconditional deployment,
  validation-only selection, and a bridge plug-in violate 38/80, 18/60, and
  7/47 estimable external diagnostics at the primary contract. Strict MOSAIC
  deploys 20/100 with 0/20 observed violations and abstains on all 20
  missing-support Camelyon jobs. These v2 outcomes are transparently labeled
  post-audit corrective evidence; deterministic, exact-rational, and
  correction-scope audits all pass.
- A fresh paired real-feature study evaluates 325 official-method rows across
  Waterbirds, Camelyon17-WILDS, CivilComments-WILDS, BiasBios, and GaitPDB.
  Transform-exact is pointwise no worse on every row and strictly improves 135,
  adding one externally safe Waterbirds deployment.
- All six externally estimable exact selections pass their natural diagnostic;
  unsupported and utility-limited cases are reported as abstentions or
  unestimable, not safe. An independent replay of 650 global optima has zero
  mismatch, and all ten preregistered real-feature gates pass.
- The repository-wide test run passes 255 tests plus 14 subtests.

These empirical statements are scoped to their locked studies. Favorable
benchmark behavior does not prove that a deployment population belongs to the
paper's structured shift class.

## Start Here

- Final release PDFs: `output/pdf/`
- Main paper: `research/maintrack/mosaic_aaai2027/mosaic_aaai2027_named.pdf`
- Anonymous paper: `research/maintrack/mosaic_aaai2027/mosaic_aaai2027_anonymous.pdf`
- Supplement: `research/maintrack/mosaic_aaai2027/mosaic_aaai2027_supplement_named.pdf`
- Reproducibility checklist: `research/maintrack/mosaic_aaai2027/mosaic_aaai2027_reproducibility_checklist.pdf`
- Anonymous code/data ZIP: `research/maintrack/mosaic_aaai2027/mosaic_aaai2027_code_data_anonymous.zip`
- Claim ledger: `research/mosaic/MOSAIC_CLAIM_LEDGER.md`
- Current theorem record: `research/mosaic/PRE_RELEASE_SHIFT_THEOREMS.md`
- Novelty collision audit: `research/mosaic/MOSAIC_NOVELTY_COLLISION_AUDIT_2026-07-18.md`
- Numerical audit erratum: `research/mosaic/TRANSFORM_EXACT_AUDIT_ERRATUM.md`
- Fresh paired real confirmation: `research/artifacts/mosaic_real_exact_confirmation_manifest_v1.json`
- Fresh paired real audit: `research/artifacts/mosaic_real_exact_confirmation_audit_v1.json`
- Fresh paired real summary: `research/artifacts/mosaic_real_exact_confirmation_summary_v1.json`
- Data-certified bridge summary: `research/artifacts/mosaic_bridge_evidence_summary_v2.json`
- Strict bridge replay audit: `research/artifacts/mosaic_bridge_strict_v2_audit_v1.json`
- Exact-rational bridge audit: `research/artifacts/mosaic_bridge_rational_v2_audit_v1.json`
- Bridge correction-scope audit: `research/artifacts/mosaic_bridge_strict_correction_v2_audit_v1.json`
- Exploratory real exact analysis: `research/artifacts/mosaic_real_transform_exact_exploratory_v1.json`
- Exploratory real exact audit: `research/artifacts/mosaic_real_transform_exact_exploratory_audit_v1.json`

## Installable API

Install the certificate and use the same bridge and global optimizer exercised
by the research pipeline:

```bash
python -m pip install .
```

The public deployment path has three steps. Construction, reference, and bridge
rows must be disjoint; protected source labels are certification inputs only.
One object certifies one registered tokenizer and configuration. Selecting among
several separately fitted objects requires allocating the familywise confidence
budget across those objects before looking at their certificates.

```python
from mosaic_certified_release import Mosaic

model = Mosaic().fit(construction_x, construction_y)
certificate = model.certify(reference_x, reference_y, reference_source,
                            bridge_x, bridge_y, bridge_source)
response = model.release_or_abstain(item_id, release_x)
```

`response` is either a persistent released token with its task prediction or
an explicit `ABSTAIN`. Repeating an item identifier returns the same sampled
token, preventing repeated-query composition from silently changing the
registered release mechanism.

## Registered LLM Extension

A prewritten six-candidate pilot tested Qwen2.5-1.5B-Instruct hidden-state
interfaces on disjoint CivilComments IDs. No candidate met the fixed go rule,
so no temporal confirmation was registered and no LLM claim was added to the
main paper. The two nonconstant K=4 candidates had certified worst errors
0.492292 and 0.490117 against a required maximum of 0.49. The complete report
and its independent stopping-rule audit are retained in
`research/artifacts/mosaic_qwen_pilot_v1.json` and
`research/artifacts/mosaic_qwen_pilot_audit_v1.json`.

## Verification

Install the lightweight confirmation environment and run the complete tests:

```bash
python -m pip install -r research/mosaic/requirements-confirmation.txt
PYTHONPATH=research/mosaic:research/scripts python -m pytest research/tests -q
```

The anonymous code/data ZIP contains exact Git snapshots for the locked
synthetic and transform-exact studies, because later theorem and audit
improvements intentionally changed files covered by the original hashes. Build
and extract it, then follow its `README.md` to replay both studies without
weakening their immutable checks:

```bash
python research/maintrack/mosaic_aaai2027/build_anonymous_code_package.py
```

Verify the official-method result package after installing the separate real
environment and mounting the pinned feature/method stores recorded by the
preregistration:

```bash
python -m pip install -r research/mosaic/requirements-real.txt
python research/mosaic/run_mosaic_real_confirmation.py --verify-only
python research/mosaic/audit_mosaic_real_frontier.py \
  research/artifacts/mosaic_real_confirmation_v1/*.json \
  --output /tmp/mosaic_real_confirmation_audit.json
python research/mosaic/audit_mosaic_real_transform_exact.py \
  --output /tmp/mosaic_real_transform_exact_audit.json
python research/mosaic/run_mosaic_real_exact_confirmation.py --verify-only
python research/mosaic/audit_mosaic_real_exact_frontier.py \
  research/artifacts/mosaic_real_exact_confirmation_v1/*.json \
  --output /tmp/mosaic_real_exact_confirmation_audit.json
```

Build the papers with the official template copy included in the repository:

```bash
cd research/maintrack/mosaic_aaai2027
latexmk -pdf mosaic_aaai2027_anonymous.tex
latexmk -pdf mosaic_aaai2027_named.tex
latexmk -pdf mosaic_aaai2027_supplement_anonymous.tex
latexmk -pdf mosaic_aaai2027_supplement_named.tex
latexmk -pdf mosaic_aaai2027_reproducibility_checklist.tex
python build_anonymous_code_package.py
```

Raw third-party datasets, frozen embedding stores, virtual environments, and
external-drive-only generated arrays are excluded from GitHub. Their provenance,
versions, hashes, and compact replay receipts are retained in the locked
manifests. The checked-in synthetic receipts contain no private or human-subject
data.

## Historical Record

This repository began as VERA, a finite-candidate risk-control approach. Its
negative P0 result and immutable preregistrations remain in the history because
they explain why MOSAIC was developed instead of hiding a failed direction.
Files that explicitly identify VERA as historical evidence are not current
MOSAIC claims.

OpenAI Codex assisted extensively with ideation, literature discovery, theorem
and proof drafting, implementation, experiment orchestration, statistical
analysis, figures, and manuscript drafting. It is not an author or an
independent scientific reviewer. Any submission requires human verification of
the complete work and compliance with the venue's current disclosure policy.
