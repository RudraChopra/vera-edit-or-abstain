# MOSAIC admitted-shift stress v2

The v1 admitted-shift artifact is retained unchanged.  In one BiasBios-Clinical
job, two Bayes-attacker assignments were tied up to floating-point roundoff.
Different evaluation orders therefore selected different, equally
worst-attacker residual vertices.  The deployment decisions and all primary
stress outcomes were unchanged, but the median distance outside the direct
target confidence region depended on that tie.

Version 2 resolves every such tie by selecting the lexicographically first
assignment whose balanced accuracy is within `1e-12` of the exact enumerated
maximum.  This produces a deterministic secondary distance of `0.3678`; v1
reported `0.3593`.  The primary result remains: the direct target-table rule
violates 16 of 36 constructed bridge-admitted laws, while MOSAIC releases 20
safe cases and abstains on the remaining 16.
