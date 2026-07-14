# Upstream Baseline Reference Inventory

Generated at UTC: `2026-07-14T00:53:00.831088+00:00`
Inventory ready: `True`

## Official-Code Repositories

| Status | Baseline | Commit | Remote | Allowed claim |
| --- | --- | --- | --- | --- |
| pass | MANCE++ | `d1a260f945914fa1d1c75a71163af3cd586eb241` | https://github.com/MatanAvitan/mance.git | Official-code FARO adapter exists; Waterbirds is claim-grade, and Camelyon17 has a full no-cap claim-grade receipt. |
| pass | R-LACE | `2d9b6d03f65416172b4a2ca7f6da10e374002e5f` | https://github.com/shauli-ravfogel/rlace-icml | Official upstream code is pinned locally; current FARO tables may only claim R-LACE-style proxy stress tests until matched receipts exist. |
| pass | TaCo | `35995e44b95dc1722b03d18b9b16c5b9f8322db5` | https://github.com/fanny-jourdan/TaCo | Official upstream code is pinned locally; current FARO tables may only claim TaCo-style proxy stress tests until matched receipts exist. |
| pass | LEACE | `9f51753821316a1edacf78b52b464ab26d40e60a` | https://github.com/EleutherAI/concept-erasure | Official upstream code is pinned locally; current FARO tables may only claim LEACE-style proxy stress tests until matched receipts exist. |

## Paper-Only or Proxy Baselines

| Baseline | Evidence | Allowed claim |
| --- | --- | --- |
| SPLINCE/SPLICE | No official upstream repository is pinned in /Volumes/Backups/FARO/external as of this audit. | Only SPLINCE/SPLICE-style proxy stress-test language is allowed until an exact upstream implementation is identified and run. |
