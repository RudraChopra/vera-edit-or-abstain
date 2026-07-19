# Public benchmark protocol

Raw benchmark data, embeddings, and model checkpoints are not redistributed.
Place independently obtained upstream data under `data/real/raw/<dataset>/`,
then run `python data/real/prepare_public_data.py --dataset <name> --source
data/real/raw/<dataset> --output data/real/processed/<dataset>`.

The preparation command writes a content-addressed `manifest.json`. A manifest
matches the paper only when its SHA-256 equals the value below. Hashes cover the
processed feature-store schema and arrays, not a private path or machine name.

| Dataset name | Upstream terms and official split used | Expected processed manifest SHA-256 |
|---|---|---|
| `biasbios_clinical` | BiasBios biography terms; fixed profession split, with clinical versus other profession as task and binary gender as source | `a2f0d7cd2d67eb2cb01a05fccbd32c24519cd916ac9ceb508ca680ce1f3aa1e4` |
| `camelyon17_wilds` | Camelyon17 pathology-slide terms; official WILDS hospital split and metadata | `6d18335f5a66a174304bbafbd7ca6efca9250fb4c450d7d35ce9791a27883fb3` |
| `civilcomments_wilds` | CivilComments terms; official WILDS train, validation, and test split with identity metadata | `bfe7394a8189cfeaa12efbaf038306b0463c70d9e05508eed6f154823b0e5c10` |
| `gaitpdb` | GaitPDB upstream terms; subject-disjoint native split, binary task, and recorded source label | `8ff97ea9c25d32e58dc5551dbadfd63f12e3582b0623808713ad3a3a81c7d8bc` |
| `waterbirds` | Waterbirds, CUB, and Places image terms; official train, validation, and test metadata split | `fd4c17004512e9b38afc086eab8544dd9a15d187323747f666f3ae38d5d8c41f` |
| `acs_income_ca_tx` | Folktables ACS 2018 one-year PUMS terms; California reference and Texas target, income above 50,000 task | `d37d371760fd813e47d08bd91baea49dc66ac8efa2bbbb0bdcdd789ea2549ae3` |

## Shared feature-store schema

Each processed store contains NumPy arrays for features `x`, task label `y`,
source label `s`, and split. ACS also retains geography metadata only for split
construction. The source attribute is excluded from the released feature
matrix. Train-only standardization and PCA produce at most 128 dimensions.
The tokenizer is a task-score logistic model fitted only on construction data,
then divided at construction-score quartiles into four fine tokens.

Certification sampling is balanced over represented source-label strata. Real
confirmation caps are 8,000 eraser-training rows, 2,000 tokenizer-construction
rows, 8,000 certification rows, and 8,000 external diagnostic rows. The ACS
diagnostic cap is 12,000. Missing required source-label strata are preserved as
missing and force abstention; they are never imputed.

## Retrieval

`download_public_data.py` invokes the standard dataset-package downloader for
WILDS and Folktables when those optional packages are installed. BiasBios,
GaitPDB, and Waterbirds require the reviewer to obtain the upstream release
under its own terms and pass the resulting directory with `--source`. No raw
data are embedded in this archive.
