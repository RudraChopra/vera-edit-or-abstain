# MOSAIC AAAI 2027 V10 Release

Release date: 2026-07-21

Use `MOSAIC_AAAI2027_V10_ANONYMOUS.pdf` with
`MOSAIC_AAAI2027_V10_SUPPLEMENT_ANONYMOUS.pdf` for double-blind review. Upload
`MOSAIC_AAAI2027_V10_REPRODUCIBILITY_CHECKLIST.pdf` and
`MOSAIC_AAAI2027_V10_CODE_DATA_SUPPLEMENT.zip` in their corresponding fields.

## Changes since V9

- Added an installable `mosaic-certified-release` package with a three-step
  `fit`, `certify`, and `release_or_abstain` interface.
- Added persistent one-token-per-item release state, serialization, and
  fail-closed recertification tests.
- Ran the prewritten six-candidate Qwen2.5-1.5B-Instruct pilot. No candidate met
  its fixed go rule, so temporal confirmation was not registered and the main
  paper was not changed. The supplement and archive preserve the complete
  negative report and independent stopping-rule audit.
- Expanded the repository-wide test run to 255 tests plus 14 subtests.

## Verification

- The anonymous main paper has seven content pages; references begin on page
  eight.
- The supplement has 13 pages and the reproducibility checklist has two pages.
- The V10 audit verifies page size, page limits, embedded fonts, metadata,
  compilation logs, archive checksums and integrity, and double-blind identity
  markers.
