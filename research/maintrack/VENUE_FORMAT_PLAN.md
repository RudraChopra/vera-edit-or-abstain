# VERA Venue Format Plan

Date: July 13, 2026

## Target Order

1. ICLR main track first, if the 2027 author guide and style files are released
   in time.
2. NeurIPS or ICML next, using the corresponding current official style file.
3. AAAI-27 as the nearest concrete AAAI target, with the official AAAI-27 author
   instructions.

## Current Verified Guidance

- ICLR 2026 author guidance reports a 9-page-or-fewer main-text limit at
  submission, with references outside the page limit. The 2027 guide must be
  rechecked before submission.
- AAAI-27 official pages are live. AAAI technical papers use a 7-page technical
  content limit excluding references, with the official AAAI author kit/style
  required. The official kit was downloaded from `https://aaai.org/authorkit27/`
  into `research/maintrack/aaai2027_template/AuthorKit27/`.
- NeurIPS 2026 official conference and handbook pages are live, but the
  checked July 13, 2026 style-file guesses
  `https://media.neurips.cc/Conferences/NeurIPS2026/Styles/neurips_2026.sty`
  and `.tex` returned HTTP 404. Do not treat a NeurIPS draft as official until
  the current style files are downloaded from the official NeurIPS style page.
- Local venue-template references are useful for drafting, but they are not a
  substitute for the official current style files.

## Current VERA Manuscript Status

`research/maintrack/faro_main.tex` is a readable generic LaTeX content draft.
`research/maintrack/iclr2026_template/iclr2026/faro_iclr2026_draft.tex` is an
anonymous ICLR-2026-style draft compiled with the official ICLR 2026 style files
downloaded from the ICLR Master-Template repository. The compiled PDF is
`research/maintrack/faro_iclr2026_draft.pdf`.

`research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_draft.tex` is
an anonymous AAAI-27-style source draft using the official `aaai2027.sty`.
Compilation is currently blocked locally because the AAAI-27 style enforces
PDFLaTeX and this Mac does not have `pdflatex`; Tectonic stops at the official
`pdfTeX is required` guard. Do not modify the official style file to bypass
this. Install a PDFLaTeX distribution before treating the AAAI draft as
compiled.

ICLR 2027 style files are not yet the verified target. Before a real
submission, replace the 2026 style with the current official target-year style.

`research/maintrack/faro_neurips_draft.tex` is a portable two-column NeurIPS-like
content draft and now matches the 80k Camelyon MANCE++ diagnostic text, but it
is not an official current NeurIPS-style submission source.

## Required Conversion Work

- Select the final target venue and replace the ICLR-2026 draft style with the
  official current target-year style files.
- For AAAI-27, install PDFLaTeX and compile
  `research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_draft.tex`
  without modifying `aaai2027.sty`.
- Convert the content draft into that exact target template.
- Keep the anonymous submission version separate from any camera-ready version.
- Move long derivations, additional ablations, diagnostic artifacts, and full
  audit tables into an appendix or supplement.
- Re-render and inspect the PDF after conversion.

## Claim Boundary

The current manuscript can support internal review and continued development,
but it should not be submitted until the official venue template, page limit,
anonymization rules, checklist, and supplement rules are satisfied.
