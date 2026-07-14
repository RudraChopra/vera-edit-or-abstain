# VERA Theory Target

## Setup

Let `P` be the validation distribution and let `Q` be an unknown deployment
distribution. Candidate edit `e` and the identity edit are evaluated on the
same example. The paired target-harm variable is

`H_e = target_loss(e(Z), Y) - target_loss(Z, Y)`.

For attacker `a`, the leakage variable `L_e,a` is its balanced correctness or
another preregistered bounded recovery score on `e(Z)`. Attackers are trained
without the certification fold. All variables and edit candidates used by a
claim-grade run are fixed by the preregistration.

The primary shift class is

`Q_Gamma(P) = {Q << P : 0 <= dQ/dP <= Gamma}`,

with `Gamma >= 1`. This expresses bounded deployment reweighting without
requiring environment labels or a single known target density ratio.

## Lemma A: Robust Paired Audit Bound

For a bounded variable `V in [a,b]`, define

`R_Gamma(V;P) = sup_{Q in Q_Gamma(P)} E_Q[V]`.

The population quantity equals upper-tail CVaR under `P`. A DKW event gives a
simultaneous finite-sample upper bound on this quantity for every preregistered
edit and attacker. This lemma is a standard concentration/DRO building block,
not the novelty claim. Multiplicity is charged across edits, contracts, source
classes, and attackers.

The proof must state all constants, support multiclass balanced leakage through
class-conditional certification, and keep attacker training independent of the
certification fold.

For the actual zero-one audits, use the exact discrete specialization rather
than discarding structure with a range-only bound. Bernoulli attacker
correctness uses a one-sided Clopper-Pearson upper bound. Paired harm in
`{-1,0,1}` uses simultaneous upper/lower binomial bounds for its positive and
negative masses and the closed-form worst reweighting allocation. The shared
confidence event is independent of `Gamma`, so the exact curves remain uniform
over the full radius.

## Theorem B: Certified Erasure Shift Radius

For edit `e`, define its population shift radius as the largest `Gamma >= 1`
for which paired target harm is at most `tau` and every registered leakage
contract is at most `lambda` over the entire ambiguity class. If the IID
contract itself fails, define the radius as zero.

VERA inverts the simultaneous robust-risk bands to return a lower confidence
bound on that common radius. With probability at least `1-delta`, every
reported edit radius is no larger than its population radius, simultaneously
over candidates and over the continuum of deployment budgets. Therefore the
user may choose any deployment budget no larger than the reported radius after
seeing the certificate without another multiplicity penalty. The proof must
also establish monotonicity, the piecewise empirical computation, and the
right-censoring convention at the declared numerical cap.

## Corollary C: Worst-Group Mixture Shift

When validated groups `g` have stable conditional distributions and deployment
may use any mixture over those groups, simultaneous per-group upper bounds
imply the same target and leakage contracts for every mixture. This is a
corollary and is not presented as the primary novelty.

## Theorem D: Support-Mismatch Impossibility

If deployment may put positive mass outside validation support, two worlds can
induce exactly the same validation observations but opposite edit outcomes on
the unsupported region. Therefore no protocol can both accept nontrivially and
uniformly control false acceptance over that unrestricted shift class. The
proof must formalize the indistinguishable worlds and connect the result to the
preregistered Camelyon17 center-2 support mismatch without claiming that the
benchmark alone proves the theorem.

## Corollary E: False Acceptance

With simultaneous confidence level `1-delta`, the probability that VERA reports
a radius covering a deployment budget at which any declared contract fails is
at most `delta`.

## Required Comparators

The experiments must compare point selection, an IID LTT certificate, and the
new ambiguity-robust paired certificate. This isolates the contribution from
both ordinary validation selection and existing same-distribution risk control.

## Non-Claim

No finite attacker portfolio proves erasure against every measurable recovery
algorithm. VERA certifies only the preregistered attacker class. A separate
impossibility or limitation statement must make this boundary explicit.
