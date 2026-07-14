# VERA Novelty Audit, 2026-07-13

## Scope

This audit asks whether a finite candidate set, simultaneous confidence bounds,
thresholded acceptance, and abstention are novel enough for an AAAI or ICLR
main-track paper on representation erasure. The search covered concept erasure,
distribution-free risk control, covariate-shift risk control, robust model
evaluation, fair-representation certificates, and impossibility under shift.
The search was performed on July 13, 2026. It prioritized primary conference
papers and arXiv records. Search phrases included `Learn Then Test covariate
shift`, `worst-group risk mixture shift`, `certified representation fairness`,
`concept erasure multiple adversaries`, and `distributionally robust model
evaluation f-divergence`.

## Decision

The original VERA mechanism is not a sufficient top-conference novelty claim.
Learn Then Test already calibrates a finite family by testing risk constraints,
and Conformal Risk Control already treats monotone risks and discusses
distribution shift. Weighted and high-probability extensions cover covariate
shift. A group-wise Hoeffding bound followed by a union bound is useful, but is
not a defensible headline theorem by itself.

The project therefore changes its primary object. VERA now means **Verified
Erasure under Reweighting Ambiguity**. It studies the incremental, paired harm
of editing a representation instead of the absolute performance of an edited
model. It seeks a certificate that holds simultaneously for every deployment
distribution whose density ratio relative to validation is bounded, without
requiring environment labels or one known target weighting function. The
worst-group mixture result is retained as a transparent special case. Leakage
is audited against a heterogeneous, preregistered attacker portfolio because
protection against one nonlinear probe is known not to transfer reliably to
another.

This is a candidate contribution, not a claim that novelty has been proved by
search. The final novelty claim remains conditional on a broader citation sweep
and external expert review.

## Closest Prior Work And Consequences

1. Angelopoulos et al., *Learn Then Test: Calibrating Predictive Algorithms to
   Achieve Risk Control*, arXiv:2110.01052 and Annals of Applied Statistics
   (2025). This invalidates any claim that finite-family testing, acceptance,
   and abstention are themselves new.
   https://arxiv.org/abs/2110.01052

2. Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets*, JACM
   68(6), 2021. This is foundational risk-control work and must be cited.
   https://doi.org/10.1145/3478535

3. Angelopoulos et al., *Conformal Risk Control*, ICLR 2024. The paper controls
   general monotone losses and includes a distribution-shift extension, so a
   generic shift-aware risk-control claim is not new.
   https://openreview.net/forum?id=33XGfHLtZg

4. Tibshirani et al., *Conformal Prediction Under Covariate Shift*, NeurIPS
   2019. Known or accurately estimated likelihood ratios can restore weighted
   validity under covariate shift.
   https://arxiv.org/abs/1904.06019

5. Almeida et al., *High Probability Risk Control Under Covariate Shift*,
   COPA/PMLR 2025. This directly extends LTT with importance-weighted
   calibration losses.
   https://proceedings.mlr.press/v266/almeida25a.html

6. Ai and Ren, *Not All Distributional Shifts Are Equal: Fine-Grained Robust
   Conformal Inference*, ICML 2024. This combines identifiable reweighting with
   worst-case conditional shift in an f-divergence ball.
   https://proceedings.mlr.press/v235/ai24a.html

7. Najafi et al., *Certifiably Robust Model Evaluation in Federated Learning
   under Meta-Distributional Shifts*, ICML 2025. This gives worst-case uniform
   evaluation guarantees under Wasserstein and f-divergence shifts.
   https://proceedings.mlr.press/v267/najafi25a.html

8. Klivans et al., *Testable Learning with Distribution Shift*, COLT 2024.
   Certify-or-reject learning under test-distribution access is already a
   developed theoretical model.
   https://proceedings.mlr.press/v247/klivans24a.html

9. Ben-David et al., *Impossibility Theorems for Domain Adaptation*, AISTATS
   2010. Any new impossibility statement must be positioned as a
   representation-intervention result, not as the first impossibility under
   distribution shift.
   https://proceedings.mlr.press/v9/david10a.html

10. Jovanovic et al., *FARE: Provably Fair Representation Learning with
    Practical Certificates*, ICML 2023. FARE already gives finite-sample upper
    certificates on downstream unfairness for restricted encoders, so VERA
    must not claim the first certified fair or erased representation.
    https://proceedings.mlr.press/v202/jovanovic23a.html

11. Deka and Sutherland, *MMD-B-Fair: Learning Fair Representations with
    Statistical Testing*, AISTATS 2023. Kernel testing for hiding protected
    information while retaining target utility is established.
    https://proceedings.mlr.press/v206/deka23a.html

12. Ravfogel et al., *Kernelized Concept Erasure*, EMNLP 2022. This work finds
    that erasure against one kernel or one nonlinear adversary often fails to
    transfer to another, and that even convex kernel combinations did not solve
    the problem. This motivates, but also constrains, VERA's attacker audit.
    https://arxiv.org/abs/2201.12191

13. Chowdhury et al., *Fundamental Limits of Perfect Concept Erasure*, AISTATS
    2025. Information-theoretic limits on erasure and utility are already
    established and must anchor VERA's limitations.
    https://proceedings.mlr.press/v258/chowdhury25a.html

14. Ravfogel et al., *Null It Out*, ACL 2020; Ravfogel et al., *Linear
    Adversarial Concept Erasure*, ICML 2022; Belrose et al., *LEACE*, 2023;
    Jourdan et al., *TaCo*, 2023; Holstege et al., *SPLINCE*, NeurIPS 2025; and
    Avitan et al., *MANCE*, 2026 define the eraser frontier against which VERA
    must be evaluated.

## Remaining Novelty Risk

Bounded-density-ratio robust risk is a standard distributionally robust
optimization object, and paired treatment-effect analyses are also standard.
The defensible claim must therefore be the complete representation-editing
problem and its evidence: paired edit-versus-identity contracts, simultaneous
target-harm and heterogeneous-attacker leakage certificates, a useful
finite-sample abstention rule, and an impossibility boundary for unsupported
deployment mass. The paper must present this as an extension and synthesis of
risk control for representation intervention, not as invention of risk control
or distributional robustness.

## Second-Pass Red Team: DKW/CVaR Is Not The Headline

A targeted second search found that finite-sample CVaR concentration through
DKW is itself established. Thomas and Learned-Miller derive concentration
bounds for CVaR, and Budde et al. explicitly use DKW confidence bands for CVaR
in statistical model checking. Najafi et al. provide robust DKW-style model
evaluation under meta-distributional shifts. These results rule out presenting
the robust-risk upper bound as a new theorem.

15. Thomas and Learned-Miller, *Concentration Inequalities for Conditional
    Value at Risk*, ICML 2019.
    https://proceedings.mlr.press/v97/thomas19a.html

16. Budde et al., *Statistical Model Checking Beyond Means: Quantiles, CVaR,
    and the DKW Inequality*, 2025.
    https://arxiv.org/abs/2509.11859

17. Jeong and Namkoong, *Robust Causal Inference under Covariate Shift via
    Worst-Case Subpopulation Treatment Effects*, COLT 2020. This is especially
    close to the paired-harm robust-risk interpretation and must be discussed.
    https://proceedings.mlr.press/v125/jeong20a.html

The revised candidate novelty is the **erasure shift radius**: the maximum
common bounded-reweighting budget under which one edit simultaneously satisfies
paired target-harm and all registered leakage contracts. VERA returns a
simultaneous lower confidence bound on this radius for every edit, valid over a
continuum of deployment budgets, and reports which contract limits the radius.
The unsupported-mass theorem explains when this radius cannot be nontrivial.
This remains a candidate domain-specific contribution, not proof of novelty;
the cold external reviews are still mandatory.
