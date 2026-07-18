# Transform-exact confirmation v1 audit erratum

The locked v1 confirmation completed all 5,000 paired tables and passed every
pre-audit gate. Its independent replay reproduced all 20,000 privacy
certificates, 40,000 utility certificates, 20,000 external risks, coverage
events, aggregate intervals, and pointwise dominance comparisons. It reported
two decision mismatches among 10,000 method rows, so the v1 audit is preserved
as failed rather than relabeled.

Both mismatches were capacity-transfer deployments at sample size 1,000. Their
largest certified privacy advantages exceeded the registered threshold by
`4.1221437379062035e-10` and `4.402583575258134e-10`. These values are below the
optimizer's `1e-9` feasibility tolerance and `2e-7` post-hoc acceptance
tolerance, but above the replay decision tolerance of `1e-10`.

The v2 correction is deliberately stricter: the deployment layer explicitly
rechecks every returned privacy certificate at threshold plus `1e-10`, matching
the independent replay. Before the v2 rerun, we predict that exactly the two
identified rows will change from deployment to abstention. The theorem,
optimizer, data-generating process, seeds, thresholds, pass conditions, and
external-safety labels are unchanged. The v1 result, failed audit, v2 amendment,
and v2 result remain available as separate, hash-linked artifacts.
