# Double-Blind Release Boundary

The named GitHub repository and the anonymous review package are different
release channels. A file being de-identified internally does not make it safe to
publish from an author-named repository: the repository location, Git history,
and byte-identical archive can reveal authorship.

## Named Repository

The named repository may contain research code, named preprint materials when
venue policy permits them, public-data acquisition instructions, compact
nonrestricted receipts, and reproducibility documentation. Before the next push,
an authenticated human must verify the repository's visibility and the current
AAAI policy on preprints and public code.

If the repository is public during double-blind review, record that fact as a
prior public exposure. Do not claim that deleting a later commit restores blind
history. Changing repository visibility or rewriting public history requires
explicit author approval and a separate policy decision.

The named repository must not track or release:

- the final anonymous review ZIP;
- anonymous submission-mode PDFs;
- anonymous flattened upload sources;
- private OpenReview identifiers or submission manifests;
- external reviewer identities or signed review forms; or
- any artifact that maps an anonymous archive hash to the author account before
  the review policy permits disclosure.

## Anonymous Review Channel

Build the anonymous paper, supplement, checklist, and code archive under an
ignored private-review directory or an external private path. Upload them only
through the venue's review system. The package must use the flattened sources
required by `AAAI_SOURCE_FINALIZATION_SPEC.md` and must pass source, PDF,
archive, metadata, path, and content scans.

The anonymous package may contain no named GitHub URL, author or affiliation,
username, home path, external-volume path, account ID, private reviewer data, or
commit metadata that maps directly to an author identity. An anonymous archive
must not be committed to the named repository merely to demonstrate that it
exists. Store its SHA-256 in the private submission registry and the local
outcome-access ledger only after the final package is frozen.

## Existing Tracked Artifacts

The current history already contains an old anonymous ZIP and anonymous PDFs.
The named branch should retain local copies if needed but remove these artifacts
from the named branch's tracked file set going forward. This does not erase prior
history and is not represented as doing so. The final visibility and exposure
decision remains a human venue-compliance gate.

## Pass Condition

1. The final anonymous package exists only in the private review channel.
2. The named branch tracks no final anonymous ZIP, submission-mode PDF, or
   flattened anonymous upload source.
3. The authenticated repository visibility and official preprint/code policy
   are human-confirmed.
4. The anonymous package passes content and metadata scans from a clean clone.
5. Any prior public exposure is disclosed to the author rather than hidden by a
   misleading `anonymity passed` audit.
