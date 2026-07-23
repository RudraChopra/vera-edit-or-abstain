import Std

namespace MosaicFormal

/-!
This file machine-checks the two algebraic soundness steps used by MOSAIC.
Probability concentration and linear-program optimality remain analytic
results in the paper; the formal statements below cover the adaptive
post-selection implication and the bridge residual construction.
-/

section AdaptiveEnvelope

variable {Law Channel Audit Score : Type}
variable [LE Score]

/-- A confidence region represented only by its membership predicate. -/
structure ConfidenceRegion (Law : Type) where
  contains : Law → Prop

/-- Every law in the region is bounded, pointwise over the channel class. -/
def IsEnvelope
    (region : ConfidenceRegion Law)
    (score : Law → Channel → Score)
    (upper : Channel → Score) : Prop :=
  ∀ law, region.contains law → ∀ channel, score law channel ≤ upper channel

/--
Theorem 1 core: a pointwise envelope remains sound after the release channel
is selected from the same audit object. No independence between `choose` and
the confidence table is used.
-/
theorem adaptive_exact_envelope_sound
    (region : ConfidenceRegion Law)
    (score : Law → Channel → Score)
    (upper : Channel → Score)
    (truth : Law)
    (truthCovered : region.contains truth)
    (envelope : IsEnvelope region score upper)
    (choose : Audit → Channel)
    (audit : Audit) :
    score truth (choose audit) ≤ upper (choose audit) :=
  envelope truth truthCovered (choose audit)

/--
The conclusion also holds for every member of a finite or infinite family of
same-table selectors on the same confidence event.
-/
theorem adaptive_family_envelope_sound
    {Selector : Type}
    (region : ConfidenceRegion Law)
    (score : Law → Channel → Score)
    (upper : Channel → Score)
    (truth : Law)
    (truthCovered : region.contains truth)
    (envelope : IsEnvelope region score upper)
    (choose : Selector → Audit → Channel)
    (audit : Audit) :
    ∀ selector, score truth (choose selector audit) ≤ upper (choose selector audit) :=
  fun selector ↦ envelope truth truthCovered (choose selector audit)

end AdaptiveEnvelope

section BridgeResidual

variable {Output Audit Witness : Type}

/-- Coordinatewise gap left after the retained common component. -/
def residualGap
    (target transported : Output → Nat) (output : Output) : Nat :=
  target output - transported output

/--
Theorem 2 algebraic core: coordinatewise bridge feasibility constructs a
nonnegative residual and reconstructs every target coordinate exactly.
-/
theorem bridge_residual_reconstruct
    (target transported : Output → Nat)
    (dominates : ∀ output, transported output ≤ target output) :
    ∀ output,
      transported output + residualGap target transported output =
        target output :=
  fun output ↦ Nat.add_sub_of_le (dominates output)

/--
The same-table bridge maximizer is covered because feasibility is pointwise
over every witness, including the selected one.
-/
theorem adaptive_bridge_residual_reconstruct
    (target : Output → Nat)
    (transported : Witness → Output → Nat)
    (feasible : ∀ witness output, transported witness output ≤ target output)
    (choose : Audit → Witness)
    (audit : Audit) :
    ∀ output,
      transported (choose audit) output
          + residualGap target (transported (choose audit)) output =
        target output :=
  bridge_residual_reconstruct
    target
    (transported (choose audit))
    (feasible (choose audit))

end BridgeResidual

end MosaicFormal
