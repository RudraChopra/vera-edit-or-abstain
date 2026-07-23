# MOSAIC Extensions: Lower Bounds, Sessions, Monitoring, and Continuous Releases

This document states the extension results used by the path-to-9 study.
All logarithms are natural. The claims here are conditional on the registered
sampling and shift assumptions; the executable checks test their finite
implementations, not the logical validity of the proofs.

## 1. A margin-dependent certification lower bound

Consider two sources with equal prior. Source zero emits a binary token from
`Bernoulli(1/2)`, while source one emits from `Bernoulli(1/2+theta)`. The
normalized Bayes source advantage is exactly `theta`.

**Theorem 1 (binary embedded lower bound).** Fix a contract `tau`, margin
`gamma`, soundness error `delta`, and power error `beta`, with
`0 < gamma < min(tau, 1/2-tau)` and `delta+beta < 1/2`. Any procedure that
certifies the law at `theta=tau-gamma` with probability at least `1-beta` and
certifies the law at `theta=tau+gamma` with probability at most `delta` needs
at least

```
log(1 / (2(delta+beta)))
----------------------------------------------
KL(Ber(1/2+tau-gamma) || Ber(1/2+tau+gamma))
```

observations from the informative source stratum.

**Proof.** Use the certification decision as a test between the safe and unsafe
experiments. Its type-I and type-II errors sum to at most `delta+beta`. For `n`
i.i.d. observations, the Bretagnolle--Huber inequality lower-bounds that sum by
`(1/2) exp(-n KL(P_safe||P_unsafe))`. Rearrangement gives the display. The
uninformative source has the same law under both hypotheses and contributes
zero KL. QED.

For fixed `tau` bounded away from `1/2`, the Bernoulli KL is `Theta(gamma^2)`;
the necessary rate is therefore
`Omega(log(1/(delta+beta))/gamma^2)`. The existing Weissman construction needs

```
n >= 2 log((2^K-2)/delta) / gamma^2
```

to make one `K`-category L1 radius at most `gamma`, matching the lower bound in
margin and confidence. Its linear dependence on `K` is also the minimax order
for a protocol that must uniformly estimate an unrestricted categorical law
in L1 before arbitrary post-selection; this is the standard multinomial L1
estimation subproblem. The binary theorem does not claim that every narrower
fixed-channel certification problem needs linear dependence on `K`.

**Corollary 1 (matching universal-region rate).** Consider protocols required
to return a distribution-uniform L1 region that remains valid for every
registered bounded functional selected after seeing the same table. The
minimax per-row sample complexity for L1 resolution `gamma` and constant error
is `Theta(K/gamma^2)`; confidence `1-delta` adds
`Theta(log(1/delta)/gamma^2)`. Consequently the Weissman region used by MOSAIC
matches the minimax rate up to universal constants in this universal-region
model.

**Justification.** The upper bound is the displayed Weissman inequality. For
the lower bound, the requested region solves unrestricted multinomial L1
estimation: the dual identity
`||p-q||_1=max_{v in {-1,1}^K} v'(p-q)` means simultaneous support control for
all bounded post-selected functionals controls L1 error. The standard
multinomial packing lower bound then yields `Omega(K/gamma^2)`, while a binary
subexperiment and Bretagnolle--Huber yield the confidence term. This
corollary is deliberately about universal post-selection regions. A test of
one scalar functional can require fewer samples.

## 2. Bounded correlated multi-item transcripts

Let one source be associated with `r` fine items
`C=(C_1,...,C_r)`. Their source-conditional law may be arbitrarily correlated.
For item `i`, a registered row-stochastic channel `M_i` releases `Z_i`.
Conditional on the complete fine tuple, the release randomizers are
independent. Define the joint channel

```
M_[r](c_1:r, z_1:r) = product_i M_i(c_i,z_i).
```

**Theorem 2 (exact bounded-session reduction).** A simultaneous confidence
region for each source's joint fine-tuple law, passed through `M_[r]`, gives the
same exact universal-attacker envelope as Theorem 1 of MOSAIC with fine
alphabet `product_i [K_i]` and output alphabet `product_i [L_i]`. This remains
valid when the joint law contains arbitrary cross-item dependence.

**Proof.** Conditional on a fine tuple, the transcript probability is exactly
the displayed product. Marginalizing the unknown joint fine law through that
matrix produces the transcript experiment observed by the attacker. MOSAIC's
support-function proof depends only on a finite source law and a row-stochastic
channel, so it applies without modification. No product assumption is made on
the source-conditional fine law. QED.

The same argument handles adaptive or correlated release randomization after
it is compiled into one registered conditional transcript channel
`Q(h|c_1:r)`. The computational cost can be exponential in `r`; the statistical
claim is exact for the registered bounded session.

For a scalable upper bound, write
`alpha_i=max_{c,c'} TV(M_i(c,.),M_i(c',.))`.

**Theorem 3 (multiplicative session capacity).** The Dobrushin coefficient of
the product channel obeys

```
alpha(M_[r]) <= 1 - product_i (1-alpha_i).
```

For `G` sources, its normalized Bayes source advantage is at most the same
right-hand side.

**Proof.** For each item, maximally couple the outputs from any two fine rows so
that disagreement probability is at most `alpha_i`. Couple item randomizers
independently. The transcript disagrees only if at least one item coupling
disagrees, whose probability is at most
`1-product_i(1-alpha_i)`. The coupling characterization of total variation
gives the Dobrushin bound. If all released source laws have pairwise TV
diameter at most `a`, choose one source as reference and use
`sum_z max_s q_s(z) <= 1 + sum_{s>1} TV(q_s,q_1) <= 1+(G-1)a`.
After balanced-prior normalization, source advantage is at most `a`. QED.

## 3. Anytime-valid recertification

For categorical observations with counts `N_n`, fix a Dirichlet density
`pi(q)`. For a candidate law `p`, define

```
E_n(p) =
  [ integral product_j q_j^(N_nj) pi(dq) ]
  / product_j p_j^(N_nj).
```

**Theorem 4 (Dirichlet-mixture confidence sequence).** If `p` is the data law,
`(E_n(p))` is a nonnegative martingale with initial value one. Therefore

```
C_n(delta) = {p : E_n(p) < 1/delta}
```

contains the true law simultaneously for every `n>=0` with probability at
least `1-delta`. The guarantee holds at arbitrary stopping times.

**Proof.** The numerator is the likelihood averaged over fixed alternative
laws. For every alternative `q`, its likelihood ratio against `p` is a
nonnegative martingale under `p`; integrating those martingales preserves that
property and initial value. Ville's inequality gives
`P_p(sup_n E_n(p) >= 1/delta) <= delta`. QED.

With Dirichlet parameter `a`, let `phat=N_n/n`. Direct algebra gives

```
log E_n(p) = log E_n(phat) + n KL(phat||p).
```

Thus the exact confidence sequence is contained in a KL ball of radius
`[log(1/delta)-log E_n(phat)]/n`; Pinsker yields the executable L1 outer radius
`sqrt(2 radius)`. Allocating `delta` across registered strata by a union bound
produces one anytime event on which every MOSAIC channel selected at every
recertification time remains covered.

## 4. Covered continuous release classes

Let `X` be a continuous internal score and let a release channel be a measurable
map `q:X -> Delta_L`. Register a class `Q` and an L1 epsilon-net
`Q_epsilon={q_1,...,q_N}` under
`sup_x ||q(x)-q_j(x)||_1 <= epsilon`.

For each source and net element, coordinate Hoeffding with a union bound gives

```
|| E q_j(X) - n^-1 sum_i q_j(X_i) ||_1
 <= L sqrt(log(2 L N / delta)/(2n)).
```

**Theorem 5 (post-selected continuous release certificate).** On the same
event, every data-selected `q in Q` obeys

```
|| E q(X) - n^-1 sum_i q(X_i) ||_1
 <= L sqrt(log(2 L N / delta)/(2n)) + 2 epsilon.
```

Passing the empirical released laws and these radii through MOSAIC's identity
channel gives a valid universal-attacker and decoder-utility certificate for
the selected continuous channel.

**Proof.** Choose a net representative `q_j` within `epsilon`. The population
output laws of `q` and `q_j` differ in L1 by at most `epsilon`, as do their
empirical averages. Add these two approximation errors to the uniform net
deviation by the triangle inequality. The resulting event is pointwise over
the complete covered class, so same-table selection is valid. Applying the
finite-output support envelope to the selected released law completes the
claim. QED.

This theorem removes the requirement that the public mechanism first expose a
hard fine token. It does not cover an unregistered continuous class with
infinite covering number, and it charges both statistical class complexity and
approximation error explicitly.

For the important class of every threshold on one real score, the finite cover
is unnecessary.

**Theorem 6 (all-threshold continuous certificate).** Let
`q_t(x)=(1{x<=t},1{x>t})` for any real `t`. With probability at least
`1-delta`, simultaneously for every `t`,

```
|| E q_t(X) - n^-1 sum_i q_t(X_i) ||_1
 <= 2 sqrt(log(2/delta)/(2n)).
```

Hence a threshold selected from the same audit sample can be passed directly
to the finite-output MOSAIC envelope with this radius, with no threshold grid
and no threshold-count correction.

**Proof.** The Dvoretzky--Kiefer--Wolfowitz inequality controls the empirical
CDF uniformly over every real `t`. A binary law's L1 distance is twice the
absolute difference in its first coordinate. The DKW event is already
pointwise over the full threshold class, so post-selection is covered. QED.

## 5. Differential real-proxy calibration

Let `R` be an imputed source, `C` the fine token, and `S` the true source.
Unlike the main paper's pooled proxy model, permit the calibrated observation
law

```
Q_y(s,c,r) = P(R=r | S=s,C=c,Y=y)
```

to vary with the token. For the latent joint law `x_y(s,c)`, the observed
proxy-token law is the linear image

```
r_y(r,c) = sum_s Q_y(s,c,r) x_y(s,c).
```

**Theorem 7 (token-dependent proxy bridge).** Suppose simultaneous confidence
events give an L1 region for each observed `r_y` and an L1 row radius for every
calibrated `Q_y(s,c,.)`. Replacing the observation matrix in the proxy
polytope by the token-dependent tensor and adding the largest calibrated row
radius to the observed-law radius contains the true `x_y` for every label.
The exact source-conditional token radii are again obtained by the same
Charnes--Cooper linear programs. Empty calibration cells force maximal
uncertainty and abstention.

**Proof.** Let `A_Q` be the block-diagonal linear map induced by `Q`.
For any probability vector `x`,
`||(A_Q-A_Qhat)x||_1` is at most the largest L1 error of a calibrated
conditional row, because it is a convex combination of those row errors.
The triangle inequality with the observed proxy-table event therefore places
`A_Qhat x` inside the enlarged observed-law region. The remaining conditional
optimization is the same linear-fractional support problem as in the pooled
proxy theorem and has the same exact Charnes--Cooper reduction. QED.

## 6. A sharp source-task conflict bound

For binary source `S` evaluated under the balanced-source audit law, normalized
Bayes source advantage is `1-2 e_S^*`, where `e_S^*` is Bayes source error.
Let the binary task label be `Y`, and write `kappa=P(Y != S)` after fixing the
registered identification of their two labels.

**Theorem 8 (privacy-utility conflict).** Every released observation and task
decoder with task error `e_Y` and source advantage at most `tau` satisfies

```
e_Y >= max(0, (1-tau)/2 - kappa).
```

The same lower bound applies to worst-label task error because it is at least
the corresponding average error. When `Y=S`, the bound is sharp.

**Proof.** Use the task decoder as a source decoder. By the union bound its
source error is at most `e_Y+kappa`, so Bayes source error is no larger.
Therefore source advantage is at least `1-2(e_Y+kappa)`. Combining this with
the contract and rearranging gives the display. For `Y=S`, a binary symmetric
channel with crossover `(1-tau)/2` attains equality. QED.

At the paper's primary `tau=.35` contract, a maximally conflicting task must
pay at least `.325` average error. This result does not say every observed
utility loss is inevitable: the data-dependent disagreement `kappa` determines
whether the lower bound is informative.
