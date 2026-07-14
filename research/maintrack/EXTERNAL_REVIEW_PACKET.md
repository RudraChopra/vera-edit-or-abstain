# VERA External Cold-Review Packet

## Purpose

Two researchers with peer-reviewed machine-learning publications should review
the same frozen anonymous package. The review gate is evidence of adversarial
feedback, not evidence that acceptance is guaranteed.

## Frozen Package

Send only these finalized artifacts, recording each SHA-256 in the private
review registry:

- anonymous main paper PDF;
- anonymous supplement PDF;
- anonymous reproduction archive.

Do not send internal readiness audits, prior critiques, response drafts, author
identity, or claimed target scores before the reviewer writes the cold review.

## Unprompted Review

Ask the reviewer to assess the work as a conference submission and identify the
strongest reasons to reject it. Request explicit comments on:

1. correctness and scope of the external-distribution guarantee;
2. novelty relative to Learn Then Test, Prompt Risk Control, conformal risk
   control, fairness certification, and concept erasure;
3. whether the unsupported-support impossibility result is meaningful;
4. whether the experiments demonstrate a need for certification;
5. whether any claim is stronger than its receipt-backed evidence;
6. the single most important missing experiment or ablation.

After the free-form review is complete, ask one binary follow-up: "Does the
paper explicitly and adequately address its overlap with Learn Then Test and
Prompt Risk Control?"

## Required Attestations

Each reviewer record must include a verifiable ML publication URL, conflict
disclosure, confirmation that the review is human-authored and cold, the exact
main-PDF hash reviewed, an explicit LTT/Prompt-Risk-Control overlap verdict,
and confirmation that every finding was transcribed into the private registry.

## Resolution

Transcribe every finding without softening it. Every critical or major item
must be fixed in the paper/code or rebutted with a concrete location. Rebuild
and re-audit the anonymous package after fixes; do not edit the original review
files.
