# FARO Main-Track Research Program

## Identity

**FARO: Frontier-Aware Representation Optimization for Target-Preserving Source Removal**

FARO is the ICLR-first reboot of the TRACE work. TRACE remains the
implementation scaffold in `research/src/trace`, but the paper identity moves
to FARO: a method and evaluation protocol for finding the safest point on a
source-removal versus target-preservation frontier. If ICLR 2027 is not ready,
the fallback sequence is ICML, NeurIPS, ICDM, AAAI, or a strong ML journal.

## Core Thesis

Robust representation learning should not blindly erase source or domain
information. When source and target evidence overlap, perfect erasure can damage
the task. FARO treats this as the central problem: estimate a frontier of
candidate edits, choose the smallest edit that reduces source leakage without
violating a target-risk constraint, and abstain when no safe edit exists.

## Main-Conference Claim

FARO is not a medical app, a hallucination detector, or a dashboard. It is a
general ML method for representation editing under distribution shift.

The intended main-paper claim is:

> Source erasure is unsafe when source and target signal overlap. FARO makes the
> tradeoff measurable, optimizes the target-preserving frontier, and exposes
> when robust repair should abstain.

The current paper contract is locked in
[`PAPER_A_LOCK.md`](PAPER_A_LOCK.md). That file is the source of truth for the
ICLR/NeurIPS/AAAI readiness target: contribution sentence, theory contract,
empirical contract, and no-go rules.

The visual story is tracked in [`FIGURE_PLAN.md`](FIGURE_PLAN.md). The main
figures must make FARO's frontier, safe set, and abstention behavior legible
without relying on prose alone.

The reproducibility story is tracked in
[`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md) and the
machine-readable manifest
[`../configs/faro_paper_a_reproducibility.json`](../configs/faro_paper_a_reproducibility.json).
These files lock the seed policy, reproduction commands, artifact map, and
claim boundaries for Paper A.

The evidence-to-claim boundary is tracked in
[`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) and
[`../configs/faro_claim_ledger.json`](../configs/faro_claim_ledger.json). This
ledger is deliberately conservative: it records what the manuscript may claim,
what it must not claim, and which artifacts prove each allowed statement.

## What Changes From TRACE

- The method identity changes from TRACE to FARO, a main-track
  representation-learning contribution.
- The key object becomes the frontier, not a single edited model.
- Abstention becomes a first-class output rather than a limitation.
- Medical data becomes one high-stakes benchmark family, not the whole project.
- Official benchmark receipts and reproducibility gates become non-negotiable.

## Current Status

As of July 13, 2026, the durable local packet has two materialized official
claim-ready rows: Waterbirds and Camelyon17-WILDS. The Waterbirds row is
scientifically useful but not a FARO win: FARO abstains under the locked
selection rule, while a group-reweighted ERM probe is stronger on worst-group
accuracy. Camelyon17-WILDS now supplies the high-stakes medical/hospital-shift
row through a complete 455,954-example frozen ResNet-18 embedding store,
NumPy-store conversion, five locked protocol rows, paired statistics, and a
passing official receipt. This is representation-reliability evidence only,
not clinical deployment evidence. CivilComments-WILDS has prior full-store
result artifacts, but the receipt and statistical report are currently macOS
File Provider dataless placeholders; the paper must not count that row until
those files are rehydrated or regenerated as durable local artifacts.

The project is therefore still not submission-ready, but the active blocker has
moved: the official benchmark breadth and high-stakes row are now present, while
the adversarial internal review and remaining claim-boundary issues must be
cleared before main-track submission.
