# VERA Claim Ledger

## Purpose

This ledger is the submission-facing contract between the manuscript and the
evidence. It separates claims that the current artifacts support from claims
that must not appear in a NeurIPS, AAAI, ICLR, ICML, ICDM, or journal
submission. The machine-readable source is
`research/configs/faro_claim_ledger.json`, and the validator is
`research/scripts/audit_claim_ledger.py`.

## Allowed Claims

VERA may claim that it reframes source or concept removal as certified
edit-or-abstain selection over a leakage-utility frontier. The evidence is the
novelty lock, algorithm specification, theory target, and manuscript. This is a
method claim about the object being optimized and the certified decision rule,
not a claim that VERA is a universally stronger eraser.

VERA may claim that the method is fully specified by a candidate edit family, a
frontier, a certified safe set, a lexicographic selection rule, and an
abstention report. The evidence is the algorithm specification and manuscript.

VERA may claim theory support for target preservation and abstention inside the
tested family. The manuscript includes rank-one target-preservation, rank-one
frontier abstention, finite-candidate safety, and finite-candidate abstention
statements, plus a simultaneous-frontier false-acceptance control corollary.
These theorems do not establish clinical safety, fairness guarantees, or
optimality outside the candidate family.

VERA may claim that Waterbirds and Camelyon17-WILDS are the current durable
official claim-ready benchmark families in the local packet. Waterbirds is an
official abstention and failure-analysis row, where VERA abstains and
group-reweighted ERM is stronger. Waterbirds must not be described as a VERA
win. Camelyon17-WILDS is an official high-stakes frozen-embedding benchmark
row with an explicit non-clinical-deployment boundary. CivilComments-WILDS may
be discussed only as prior non-durable stress evidence until its full-store
receipt and paired statistical report are regenerated or rehydrated as
materialized local artifacts.

VERA may claim that abstention is demonstrated by a synthetic overlap
certificate, a prior CivilComments-WILDS frontier stress test, and a full
Camelyon17-WILDS projection-frontier certificate. In the Camelyon17 certificate,
selection uses validation metrics only; the tested source-direction projection
family has zero certified source-reduction lower bound at every strength, so
VERA returns `ABSTAIN`. External source leakage is scoped out for that
certificate because the binary source encoding has a single source class in the
external split.

VERA may claim that MANCE++ has a full official-code reference baseline on
Waterbirds with five seeds and claim-grade statistics, plus a full no-cap
claim-grade Camelyon17-WILDS reference row under the frozen-representation
protocol. The upstream inventory pins official repositories for MANCE++,
R-LACE, TaCo, and LEACE. The current SPLINCE/SPLICE, R-LACE, TaCo, and LEACE
rows remain matched proxy stress tests unless separate reference-implementation
receipts are added.

VERA may claim that the reproducibility packet has locked seeds, commands, an
artifact map, release boundaries, and two durable claim-ready official rows.

## Forbidden Claims

Do not claim that VERA beats group-reweighted ERM on Waterbirds. Do not claim
that VERA is clinically safe or deployment-ready. Do not claim universal
state-of-the-art performance across domain-generalization or concept-erasure
benchmarks. Do not claim that VERA works without source or environment labels.
Do not treat the Camelyon17-WILDS frozen-embedding benchmark as clinical
deployment evidence. Do not call the current proxy baseline rows official
reference implementations.

## Current Submission Boundary

The local packet now has two durable official rows, a high-stakes official
Camelyon17-WILDS row, a claim-grade Waterbirds MANCE++ reference row, a large
full no-cap Camelyon17 MANCE++ reference row, pinned official upstream
repositories for close eraser baselines, and a green adversarial internal
review. The paper remains bounded as a protocol contribution rather than a
universal erasure
state-of-the-art claim. The claim ledger is meant to make that boundary
difficult to accidentally erase while drafting.
