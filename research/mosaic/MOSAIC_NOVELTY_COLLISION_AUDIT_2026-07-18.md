# MOSAIC novelty-collision audit

Date: 2026-07-18; expanded 2026-07-19 for the data-certified bridge

## Claim under audit

MOSAIC does not claim stochastic mappings, finite-sample fairness certificates,
distributionally robust optimization, contraction coefficients, or
accept-or-abstain testing individually. The narrow claim is the combined
construction:

1. one multinomial confidence event on pre-release fine-token laws;
2. uniform same-table coverage of a continuum of source-blind stochastic
   release channels and task decoders;
3. exact balanced Bayes risk for every downstream released-token attacker;
4. an exact external supremum under a common-transform polytope plus bounded
   source-specific contamination;
5. matching worst-stratum task utility and a globally solved finite-alphabet
   release-or-abstain rule;
6. a finite-sample robust LP that learns one common transform and the largest
   uniformly certifiable retained mass from labeled target bridge data; and
7. persistent-release semantics, with exact product-channel certification for
   a bounded number of fresh queries.

## Nearest neighbors checked

| Work | Genuine overlap | Remaining distinction |
|---|---|---|
| [FARE](https://arxiv.org/abs/2210.07213) | Finite restricted representations and a high-confidence certificate for any downstream classifier. | FARE certifies a restricted encoder on its reference distribution. It does not optimize a same-table stochastic post-channel under a common-transform/differential-contamination external class. |
| [Learning Smooth and Fair Representations](https://proceedings.mlr.press/v130/gitiaux21a.html) | Finite-sample fairness guarantees for randomized/smoothed representations. | Uses smoothness and chi-squared mutual information; no exact finite-alphabet Bayes envelope, transform-exact shift supremum, or joint decoder optimization. |
| [Learn Then Test](https://doi.org/10.1214/24-AOAS1998) and [Pareto Testing](https://openreview.net/forum?id=cyg2YXn_BqF) | Registered risk tests, multiplicity control over candidates, and abstention. | Tests a finite candidate family. MOSAIC covers an uncountable channel family through one sufficient-table event, then optimizes inside that event. Separate learned tokenizers still pay multiplicity. |
| [Optimal Fair Learning Robust to Adversarial Distribution Shift](https://proceedings.mlr.press/v267/agarwal25b.html) | Randomization, fairness, and robustness to malicious distribution noise. | Optimizes a fair predictor and studies robustness of its accuracy; it does not certify a public representation against every downstream attacker from a finite table. |
| [Optimized Pre-Processing for Discrimination Prevention](https://proceedings.neurips.cc/paper/2017/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html) | Stochastic preprocessing and privacy-utility tradeoffs. | Population optimization without MOSAIC's adaptive finite-sample or structured external-shift certificate. |
| [Fundamental Limits of Perfect Concept Erasure](https://openreview.net/forum?id=bppVexkY5N) | Information-theoretic privacy-utility limits for concept erasure. | Establishes erasure limits, not the confidence envelope, external shift class, or release-or-abstain optimizer. |
| [On the Contractivity of Privacy Mechanisms](https://arxiv.org/abs/1801.06255) | Dobrushin-style channel contraction. | Supplies a classical ingredient. MOSAIC's exact transform-polytope envelope is tighter than the contraction fallback and is coupled to adaptive finite-sample selection. |
| [Equivalent Comparisons of Experiments](https://doi.org/10.1214/aoms/1177729032), [Sufficiency and Approximate Sufficiency](https://doi.org/10.1214/aoms/1177700372), and [Comparison of Statistical Experiments](https://doi.org/10.1017/CBO9780511666353) | Blackwell garbling, approximate sufficiency, deficiency, and Markov-kernel comparison are classical. | MOSAIC does not claim experiment comparison as new. Its bridge is a finite-sample one-sided contamination certificate over simultaneous empirical confidence regions, composed with same-table release optimization. |
| [Weighted Garbling](https://arxiv.org/abs/2410.21694) | Generalizes Blackwell order using state-independent signal weights and characterizes conditional informativeness. | This is a close population-level neighbor and must be cited. MOSAIC instead certifies a common Markov transform plus arbitrary state-specific residual mixture from finite reference and bridge samples, maximizes uniform retained mass, and propagates that event to source leakage and task utility. |
| [Testable Learning with Distribution Shift](https://proceedings.mlr.press/v247/klivans24a.html) | Uses target data to test whether a learner can guarantee shifted-distribution performance. | Supplies the test-or-abstain paradigm, not the finite-experiment bridge LP, universal finite-token attacker, or joint stochastic release optimizer. |
| [Equivalence of Coarse and Fine-Grained Models for Learning with Distribution Shift](https://proceedings.mlr.press/v336/patel26a.html) | Proves an equivalence between pointwise PQ rejection and whole-domain testable-distribution-shift rejection in the distribution-free setting, with hardness consequences. | Concerns classifier learnability and rejection granularity. It does not estimate a common Markov bridge, bound differential contamination, or certify an adaptively selected release channel. |
| [Testing Noise Assumptions of Learning Algorithms](https://proceedings.mlr.press/v336/goel26a.html) | Tests whether data satisfy registered noise assumptions and returns a certificate of classifier optimality when the test accepts. | Tests structured label-noise assumptions for learning halfspaces. MOSAIC instead certifies finite categorical experiment comparison and propagates that bridge through a source-leakage and utility release contract. |
| [Data-Driven Robust Optimization](https://arxiv.org/abs/1401.0212) | Builds statistically calibrated uncertainty sets and optimizes decisions robustly over them. | Supplies broad robust-optimization precedent. MOSAIC's claim is the exact finite-experiment geometry and its composition with adaptive source-leakage release, not data-driven uncertainty sets in general. |

The sweep also included conformal risk control under covariate or likelihood-ratio
shift, privacy funnels, distributionally robust fair representation learning,
concept-erasure methods, conditional shift-detection martingales, and 2025--2026
OpenReview results on representation drift, Wasserstein fairness audits, and
robust erasure. None uses MOSAIC's jointly learned finite-experiment bridge and
same-table stochastic release certificate.

## Verdict

No exact collision was found for the seven-part construction above. This is a
documented negative search, not proof of priority. The strongest
collision risk is a reviewer interpreting MOSAIC as merely Learn-Then-Test over
channels, FARE with a randomized encoder, or a finite-sample wrapper around
Blackwell/Le Cam comparison. The paper addresses all three directly: it states
what is inherited, identifies the confidence-table object that removes
channel-count multiplicity, and isolates the bridge-to-release composition as
the narrow technical addition.

This audit cannot prove worldwide novelty. The paper must keep its contribution
claim narrow, cite all component literatures, and avoid claiming that
randomization, universal downstream fairness, risk-control abstention, or
Dobrushin contraction are new by themselves.
