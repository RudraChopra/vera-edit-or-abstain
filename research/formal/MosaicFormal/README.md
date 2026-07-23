# MosaicFormal

This dependency-free Lean 4 package checks two soundness steps behind MOSAIC:

1. A confidence envelope that is pointwise over the complete channel space
   remains valid when the channel is selected from the same audit table.
2. Coordinatewise bridge domination constructs a nonnegative residual gap and
   reconstructs the target law, including for a same-table selected witness.

The package does not claim to formalize concentration inequalities, linear
program duality, or floating-point solver correctness. Those are covered by
the manuscript proofs and the independent numerical audit.

Build with:

```sh
lake build
```
