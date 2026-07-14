# VERA Reviewer Packet

## What A Reviewer Should Believe

VERA is a decision protocol for representation editing. Its primary novelty is
not a new projection formula; it is the certified accept-or-abstain contract
over a leakage-utility frontier.

## Evidence Map

- Novelty: `NOVELTY_LOCK.md` and `NOVELTY_SWEEP_2026_UPDATE.md`.
- Algorithm: `ALGORITHM_SPEC.md`.
- Theory: `THEORY_TARGET.md` and the manuscript.
- Benchmarks: Waterbirds and Camelyon17-WILDS official receipts.
- MANCE++: full Waterbirds official-code reference statistics plus Camelyon17
  diagnostic boundary.
- Reproducibility: `REPRODUCIBILITY_CHECKLIST.md` and audit artifacts.
- Claim control: `CLAIM_LEDGER.md`.

## Expected Attacks

The strongest attack is that VERA is a wrapper around erasers. The response is
that the wrapper is the method: the paper studies the certified decision problem
that prior unconditional erasers do not solve.

The second attack is that Waterbirds is not a VERA win. The response is to say
so explicitly and use it as failure analysis and abstention motivation.

The third attack is MANCE++. The response is the official-code Waterbirds
baseline and a clear Camelyon17 diagnostic boundary.
