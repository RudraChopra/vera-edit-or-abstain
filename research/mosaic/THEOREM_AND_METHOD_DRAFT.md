# MOSAIC: Theorem-First Research Draft

> **Superseded negative control (July 18, 2026).** This independent
> likelihood-ratio-shift construction is not the proposed method. It fails the
> usefulness gate because independently chosen source shifts can create
> normalized leakage `1 - 1/Gamma` even when the reference laws are identical.
> It is retained to document the failed hypothesis. The active theorem is
> [PRE_RELEASE_SHIFT_THEOREMS.md](PRE_RELEASE_SHIFT_THEOREMS.md).

**Working title:** *MOSAIC: Exact Shift-Robust Certificates for Universal
Environment Erasure via Post-Selection-Safe Token Coarsening*

**Status:** research specification, July 18, 2026. This is not a result claim,
not a submission draft, and not an update to VERA. The theorem below must
survive an independent proof review and fresh preregistered experiments before
it can appear in a paper.

## The Problem We Can Actually Certify

Medical image models often retain a hospital, scanner, or stain signature. A
linear probe that fails to recover the hospital does not show that the signature
is gone: a later nonlinear probe can find it. The final VERA stress test exposed
that exact issue when an unregistered KNN probe found leakage outside the
registered portfolio.

MOSAIC changes the object being released. A frozen feature extractor maps an
image to an embedding, and a learned tokenizer maps that embedding to one of
`K` public tokens. A downstream party receives only the token. Because the
alphabet is finite, the best possible binary hospital/source attacker is known
exactly: it chooses a subset of tokens and predicts one source on that subset.
The certificate therefore covers every measurable attacker, not a shortlist of
probes.

The guarantee is deliberately conditional and modelled:

- `S in [G]` is a finite source/environment label. The main pathology study
  will certify all supported hospitals jointly and separately report
  prespecified hospital-pair diagnostics; it will not invent an undefined
  "unseen hospital" guarantee.
- `Y` is the diagnostic label. We condition on `Y` so an attacker does not win
  merely because hospitals have different disease prevalence.
- On deployment, within each fixed `(S,Y)` stratum, the token law may change
  by a declared likelihood-ratio budget `Gamma >= 1`.
- The external distribution must be conditionally absolutely continuous with
  respect to the reference stratum. MOSAIC cannot certify a source/label
  stratum that has no reference support.

This is a meaningful medical-AI contract: it says that changes in case/image
composition within a known hospital-label stratum, up to a declared severity,
cannot make any downstream token-based attacker recover the hospital beyond the
reported bound. It does not claim clinical validation, causal invariance, or
protection at a completely unsupported hospital.

## Notation

For a fixed tokenization `f: X -> [K]`, write

`p_{s,y}(c) = P(f(X)=c | S=s,Y=y)`.

For each stratum, define the bounded conditional-shift set

`W_Gamma(p) = {q in Delta_K : Gamma^{-1} p(c) <= q(c) <= Gamma p(c), all c}`.

The external family `U_Gamma(P)` contains all distributions whose conditional
token laws lie in these sets, while allowing the case-mixture `Q(S,Y)` to vary
arbitrarily. The latter is intentional: a certificate on every `(S,Y)` stratum
also covers any mixture of those strata.

For an attacker `h: [K] -> [G]`, define its conditional balanced source
accuracy at label `y` by

`BA_y(h; q_1,...,q_G) = G^{-1} sum_s q_s(h(C)=s)`.

This metric fixes the source prior to be uniform, so a source does not become
"erased" merely because it is rare. For `G=2`, the normalized universal
attacker advantage `2 sup_h BA_y(h)-1` is exactly total variation.

For `a in [0,1]`, introduce two exact event-mass functions:

`U_Gamma(a) = min{Gamma a, 1 - (1-a)/Gamma}`,

`L_Gamma(a) = max{a/Gamma, 1 - Gamma(1-a)}`.

They are the largest and smallest probability an event with reference mass `a`
can have after a `Gamma` likelihood-ratio shift.

## The Main Theorem

**Theorem 1 (MOSAIC universal multiclass shift envelope).** Let one fine
tokenizer be fixed without using the certification fold.
For group `s` and label `y`, the certification fold supplies `n_{s,y}` IID
tokens with empirical distribution `p_hat_{s,y}`. Let `epsilon_{s,y}` be any
simultaneous L1 confidence radius satisfying

`P(for all s,y: ||p_hat_{s,y} - p_{s,y}||_1 <= epsilon_{s,y}) >= 1-delta`.

For an attacker assignment `a in [G]^K`, let
`A_s(a)={c in [K]:a(c)=s}` and define

`u_{s,y}(A)=0` when `A` is empty, `u_{s,y}(A)=1` when `A=[K]`, and otherwise

`u_{s,y}(A)=min{1,p_hat_{s,y}(A)+epsilon_{s,y}/2}`.

For each label `y` and every `Gamma >= 1`, define

`A_bar_y(Gamma) = G^{-1} max_{a in [G]^K}`
` sum_s U_Gamma(u_{s,y}(A_s(a))).`

Then, with probability at least `1-delta`, simultaneously for every label `y`,
every shift budget `Gamma >= 1`, every `Q in U_Gamma(P)`, and every (including
unseen and nonlinear) downstream attacker `h: [K] -> [G]`,

`BA_y(h; Q) <= A_bar_y(Gamma).`

For two sources, an equivalent, faster total-variation form is

`D_bar_y(Gamma) = max_{A subseteq [K]} [`
` U_Gamma(u_{1,y}(A))`
` - L_Gamma(l_{0,y}(A)) ],`

where `l` is the matching structural L1 lower endpoint. Then
`BA_y(h;Q) <= [1+D_bar_y(Gamma)]/2`, and no downstream binary predictor can
have conditional demographic-parity gap above `D_bar_y(Gamma)`.

**Why this is not a finite probe theorem.** The maximization has `G^K` possible
attackers (`2^K` in the binary case) and exactly equals the supremum over all
functions on the released token. No attacker is trained, registered, or assumed
linear. The code in `mosaic_envelope.py` enumerates this maximum rather than
using a surrogate classifier.

**Proof sketch.** For fixed `q_1,...,q_G`, pointwise Bayes classification gives

`sup_h BA_y(h) = G^{-1} sum_c max_s q_s(c)`

and also `max_{a in [G]^K} G^{-1} sum_s q_s(A_s(a))`. For a fixed assignment,
the source-conditional shifts are independent, so each event mass can attain
`U_Gamma(p_s(A_s))` by a two-level reweighting. On the L1 confidence event,
every nontrivial `p_s(A_s)` is at most
`p_hat_s(A_s)+epsilon_s/2`; empty and full events remain exactly zero and one.
Monotonicity of `U_Gamma` yields the envelope. The same confidence event is
independent of `Gamma`, so the conclusion holds for the entire continuum of
budgets without a union bound over `Gamma`. The binary TV form follows from
the Neyman--Pearson/TV identity and the matching lower event endpoint.

## The New Post-Selection Result

MOSAIC starts with a fine tokenizer learned only on a development split and
constructs a hierarchy of deterministic token merges. A released representation
is any coarsening `pi: [K] -> [K']` selected after inspecting the certification
histogram. Naively, selecting a cut after seeing the certificate would demand a
large multiple-testing penalty. It does not here.

**Theorem 2 (post-selection closure under coarsening).** On the same fine-token
L1 confidence event in Theorem 1, for every deterministic coarsening `pi`,

`||pi_# p_hat_{s,y} - pi_# p_{s,y}||_1 <= epsilon_{s,y}`

simultaneously. Therefore Theorem 1 applied to the coarsened empirical
histograms, retaining the *fine* radii, is valid for every coarsening selected
as an arbitrary function of the certification data. No alpha allocation over
the number of cuts, merges, or partitions is needed.

**Proof.** Deterministic push-forward is an L1 contraction:

`sum_j |sum_{c:pi(c)=j} [p(c)-p_hat(c)]| <= sum_c |p(c)-p_hat(c)|`.

The event is already simultaneous before the selection occurs. The robust
envelope is a deterministic consequence of that event, so data-dependent
selection of `pi` cannot invalidate it.

This is the proposed paper's technical delta over a finite candidate LTT gate:
the certificate supports post-certification selection over a combinatorially
large family of token coarsenings using a single fine-token confidence object.
It is not a claim that arbitrary neural encoders can be retrained on the
certification fold.

## Exactness and the Utility Side

**Proposition 3 (conditional-set exactness).** For fixed empirical histograms
and stated L1 radii, `A_bar_y(Gamma)` is the exact supremum of robust balanced
source accuracy over every L1 confidence ball and every conditional
likelihood-ratio shift set. It is not a union-bound relaxation over attackers.
For a fixed assignment, moving `epsilon/2` probability mass into each nontrivial
correct-decision event attains the inner extrema independently by source;
maximizing over the finite assignment family attains the outer one.

A diagnostic token rule `g o pi` may be chosen together with the coarsening
after inspecting the certification histogram. For its error-token subset
`E_{s,y}(pi,g)`, the *same* fine-token L1 event gives the uniform endpoint

`e_bar_{s,y}=min{1,p_hat_{s,y}(E_{s,y})+epsilon_{s,y}/2}`

with the structural values zero and one retained for empty/full events. Hence,

`sup_{Q in U_Gamma(P)} Q[g(pi(f(X))) != Y | S=s,Y=y]`
` <= U_Gamma(e_bar_{s,y}).`

Because `Q(S,Y)` may vary arbitrarily, the maximum of those stratum bounds
controls worst-case case-mix risk. This is simultaneous over every coarsening
and token rule, so jointly selecting the representation and diagnostic rule
does not consume a new familywise error budget. A deployment is certified only
when both contracts pass.
This prevents “erasure” from winning merely by destroying diagnostic signal.

## The Necessary Impossibility Boundary

**Proposition 4 (support necessity).** If an external shift model permits
positive mass outside the reference conditional support, no nontrivial universal
source-leakage certificate is possible: an adversary can put the two source
conditional laws on different unsupported tokens, yielding total variation one.
Likewise, a missing `(S,Y)` reference stratum has no distribution-free
conditional certificate. MOSAIC must report `ABSTAIN_UNSUPPORTED`, not infer a
number, in either case.

This is an assumption boundary, not a trick to make the theorem look stronger.
External code-support monitoring can catch unconditional novel tokens, but it
cannot certify an unobserved label-conditional stratum without labels. The
paper must say that plainly.

## Algorithm: Minimax-Optimized Source-Agnostic Invariant Channels

1. Train one high-utility fine tokenizer on a development split only. The
   planned implementation uses frozen pathology embeddings, task-aware vector
   quantization, and a source-conditional penalty.
2. On the certification histogram, build a certificate-aware agglomerative
   merge path. At each step, choose the merge and token diagnostic rule that
   maximize certified utility subject to the universal source contract. This
   data-dependent search is covered by Theorem 2; the tokenizer itself remains
   frozen.
3. On the certification split, form the fine token contingency tables by
   source and label. Build their simultaneous L1 confidence radii.
4. Evaluate every hierarchy cut at every inspected `Gamma` using the exact
   multiclass envelope and the uniform utility bound, reusing the same fine
   confidence event. Select the most useful cut that passes both contracts;
   otherwise abstain.
5. On a locked external evaluation split, report ordinary and controlled
   likelihood-ratio stress tests. The external data are not part of the proof.

Coarsening is monotone for the population robust leakage envelope: a
deterministic post-processing cannot increase total variation, and it maps the
likelihood-ratio ambiguity set exactly into the corresponding coarsened set.
The actual high-confidence bound can trade tightness against token count, which
is why the hierarchy is selected empirically but certified after selection.

## Collision Matrix to Resolve Before a Paper

| Neighbor | What it already does | Required MOSAIC distinction |
| --- | --- | --- |
| FARE (ICML 2023) | Finite representations and IID certificates for every downstream classifier | MOSAIC needs the exact conditional likelihood-ratio shift envelope, all-Gamma validity, and post-certificate coarsening theorem. FARE's shift study is empirical. |
| Kang et al. (NeurIPS 2022) | Distributional fairness certification for a fixed model under a distributional-distance model | MOSAIC certifies a released representation against every downstream token attacker, uses finite-sample conditional histograms, and selects a token coarsening after inspection. |
| Learn Then Test / Pareto Testing | Finite candidate tests and abstention | MOSAIC does not claim this machinery is new. Its selection guarantee comes from one L1 confidence polytope contracting under all token coarsenings, rather than candidatewise confidence tests. |
| MMD-B-Fair | Learns fair representations by a kernel two-sample testing objective | A failed/low-power attacker test is not a universal attacker upper bound. MOSAIC's finite code makes the supremum exact. |

The collision claim is still provisional until the full primary-source review,
proof review, and baselines are complete.

## Immediate Falsification Gates

- **Math gate:** independently enumerate small token distributions and verify
  that the closed-form envelope equals the exhaustive robust optimum; test
  coarsening monotonicity and the stated confidence coverage.
- **Novelty gate:** find and read the closest work on shift-robust finite
  representations, post-selection fair-representation certificates, and
  likelihood-ratio fairness audits. If it already gives this full theorem, stop
  claiming novelty and pivot again.
- **Usefulness gate:** in a synthetic controlled-shift experiment, MOSAIC must
  retain substantially more exact-safe deployments than the old VERA envelope
  while preventing unregistered nonlinear hospital probes from exceeding its
  universal bound.
- **Medical gate:** on frozen public pathology features, evaluate at least two
  prespecified hospital pairs with a held-out hospital/source stress split. Do
  not call it clinical validation.

## References to Read and Cite

- Jovanovic et al., *FARE: Provably Fair Representation Learning with Practical
  Certificates*, ICML 2023.
- Kang et al., *Certifying Some Distributional Fairness with Subpopulation
  Decomposition*, NeurIPS 2022.
- Deka and Sutherland, *MMD-B-Fair*, AISTATS 2023.
- Angelopoulos et al., *Learn Then Test*, AOAS 2025.
- Cauchois et al., *Robust Validation*, JASA 2024.
- Lechner et al., *Impossibility Results for Fair Representation*, NeurIPS
  2021.
- Weissman et al., *Inequalities for the L1 Deviation of the Empirical
  Distribution*, 2003.
