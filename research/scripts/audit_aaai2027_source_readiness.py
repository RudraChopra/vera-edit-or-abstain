"""Audit the VERA AAAI-27 source package without modifying official style files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
AAAI_DIR = ROOT / "maintrack" / "aaai2027_template" / "AuthorKit27"
SOURCE = AAAI_DIR / "faro_aaai2027_draft.tex"
ANONYMOUS_SOURCE = AAAI_DIR / "faro_aaai2027_anonymous.tex"
NAMED_SOURCE = AAAI_DIR / "faro_aaai2027_named.tex"
STYLE = AAAI_DIR / "aaai2027.sty"

DEFAULT_JSON = ARTIFACT_DIR / "aaai2027_source_readiness.json"
DEFAULT_MD = ARTIFACT_DIR / "aaai2027_source_readiness.md"


@dataclass(frozen=True)
class Check:
    key: str
    status: str
    evidence: str
    requirement: str


def materialized_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    stat = path.stat()
    return not (stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0)


def load_text(path: Path) -> str:
    if not materialized_file(path):
        return ""
    return path.read_text(encoding="utf-8")


def status(pass_condition: bool, warn: bool = False) -> str:
    if pass_condition:
        return "pass"
    return "warn" if warn else "fail"


def estimate_main_words(source: str) -> int:
    body = re.sub(r"\\begin\{thebibliography\}.*", "", source, flags=re.S)
    body = re.sub(r"%.*", "", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", body)
    body = re.sub(r"[^A-Za-z0-9]+", " ", body)
    return len([token for token in body.split() if token])


def collect_checks() -> tuple[list[Check], dict[str, Any]]:
    source = load_text(SOURCE)
    anonymous_source = load_text(ANONYMOUS_SOURCE)
    named_source = load_text(NAMED_SOURCE)
    normalized_source = re.sub(r"\s+", " ", source)
    style = load_text(STYLE)
    pdflatex = shutil.which("pdflatex")
    word_count = estimate_main_words(source)
    stale_terms = ["40k/16k/16k", "40,000", "16,000", "$0.8474$", "$0.5410$", "$0.8759$"]
    stale_present = [term for term in stale_terms if term in source]
    anonymous_identity_terms = [
        "Rudra",
        "Chopra",
        "RudraChopra",
        "github.com",
        "Contra Costa",
        "Independent Researcher",
    ]
    anonymous_leaks = [
        term
        for term in anonymous_identity_terms
        if term.lower() in anonymous_source.lower()
    ]
    required_sections = [
        "\\section{Introduction}",
        "\\section{Method}",
        "\\section{Theory}",
        "\\section{Experiments}",
        "\\section{Reference MANCE++ Baseline}",
        "\\section{Limitations}",
        "\\section{Code Availability}",
        "\\section{Conclusion}",
    ]
    missing_sections = [section for section in required_sections if section not in source]
    checks = [
        Check(
            key="source_materialized",
            status=status(materialized_file(SOURCE)),
            evidence=f"path={SOURCE}; materialized={materialized_file(SOURCE)}",
            requirement="AAAI source must exist locally and not be a dataless placeholder.",
        ),
        Check(
            key="official_style_materialized",
            status=status(materialized_file(STYLE) and "AAAI" in style),
            evidence=f"path={STYLE}; materialized={materialized_file(STYLE)}",
            requirement="Official AAAI-27 style file must be present.",
        ),
        Check(
            key="uses_official_submission_style",
            status=status("\\usepackage[submission]{aaai2027}" in source),
            evidence="source uses \\usepackage[submission]{aaai2027}",
            requirement="Submission source must use the official AAAI submission style.",
        ),
        Check(
            key="anonymous_author_block",
            status=status(
                "\\author{Anonymous Submission}" in source
                and "VERA Project" not in source
                and "Rudra" not in source
                and "Chopra" not in source
            ),
            evidence="author block is anonymous and local identity strings are absent",
            requirement="AAAI double-anonymous source must not expose author identity.",
        ),
        Check(
            key="anonymous_source_materialized",
            status=status(materialized_file(ANONYMOUS_SOURCE)),
            evidence=f"path={ANONYMOUS_SOURCE}; materialized={materialized_file(ANONYMOUS_SOURCE)}",
            requirement="Dedicated anonymous AAAI source must exist locally.",
        ),
        Check(
            key="anonymous_source_identity_free",
            status=status(
                "\\author{Anonymous Submission}" in anonymous_source
                and "\\section{Code Availability}" in anonymous_source
                and not anonymous_leaks
            ),
            evidence=f"anonymous_leaks={anonymous_leaks}",
            requirement="Anonymous AAAI source must include code availability without named identity or GitHub URL.",
        ),
        Check(
            key="named_source_release_metadata",
            status=status(
                materialized_file(NAMED_SOURCE)
                and "\\author{Rudra Chopra}" in named_source
                and "Contra Costa County" in named_source
                and "https://github.com/RudraChopra/vera-edit-or-abstain" in named_source
                and "\\section{Code Availability}" in named_source
            ),
            evidence=(
                f"path={NAMED_SOURCE}; materialized={materialized_file(NAMED_SOURCE)}; "
                f"has_author={'Rudra Chopra' in named_source}; "
                f"has_repo={'https://github.com/RudraChopra/vera-edit-or-abstain' in named_source}"
            ),
            requirement="Named AAAI source must contain author metadata and the public release URL.",
        ),
        Check(
            key="required_sections_present",
            status=status(not missing_sections),
            evidence=f"missing_sections={missing_sections}",
            requirement="AAAI source must include the core method-paper sections.",
        ),
        Check(
            key="current_camelyon_mance_reference",
            status=status(
                "full no-cap" in normalized_source
                and "302,436/68,464/85,054" in source
                and "$0.5635$" in source
                and "$0.8741$" in source
            ),
            evidence="AAAI source contains the full no-cap Camelyon MANCE++ reference numbers",
            requirement="AAAI source must cite the full no-cap Camelyon MANCE++ receipt rather than superseded diagnostics.",
        ),
        Check(
            key="no_stale_camelyon_mance_terms",
            status=status(not stale_present),
            evidence=f"stale_terms_present={stale_present}",
            requirement="Superseded 40k Camelyon MANCE++ text must be removed.",
        ),
        Check(
            key="false_acceptance_corollary_present",
            status=status("False-acceptance control" in source),
            evidence="source contains the false-acceptance corollary",
            requirement="Theory section should include explicit false-acceptance control.",
        ),
        Check(
            key="reviewer_attack_preempted",
            status=status("strongest reviewer objection" in source.lower()),
            evidence="source explicitly preempts the eraser-versus-decision-layer objection",
            requirement="AAAI source should preempt the strongest baseline-framing attack.",
        ),
        Check(
            key="reference_boundary_present",
            status=status(
                all(term in source for term in ("R-LACE", "TaCo", "LEACE", "proxy"))
                and "state-of-the-art erasure" in source
            ),
            evidence="source separates pinned/proxy baselines and denies universal erasure SOTA",
            requirement="AAAI source must not overclaim reference parity.",
        ),
        Check(
            key="clinical_boundary_present",
            status=status(
                "not clinical evidence" in normalized_source
                and "do not establish clinical safety" in normalized_source
            ),
            evidence="source states Camelyon17/GaitPDB are not clinical deployment evidence",
            requirement="Medical benchmark language must not imply clinical deployment readiness.",
        ),
        Check(
            key="estimated_length_reasonable",
            status=status(word_count <= 5600, warn=word_count <= 6500),
            evidence=f"estimated_main_words={word_count}",
            requirement="AAAI source should remain plausibly within the 7-page technical limit.",
        ),
        Check(
            key="pdflatex_available",
            status=status(pdflatex is not None, warn=True),
            evidence=f"pdflatex={pdflatex or '<missing>'}",
            requirement="Local final AAAI PDF compilation requires PDFLaTeX.",
        ),
    ]
    metadata = {
        "source": str(SOURCE),
        "anonymous_source": str(ANONYMOUS_SOURCE),
        "named_source": str(NAMED_SOURCE),
        "style": str(STYLE),
        "estimated_main_words": word_count,
        "pdflatex": pdflatex,
        "compile_blocker": None if pdflatex else "PDFLaTeX is not installed locally",
    }
    return checks, metadata


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AAAI-27 Source Readiness Audit",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Source ready: `{report['source_ready']}`",
        f"Warning count: `{report['warn_count']}`",
        "",
        "| Status | Check | Evidence | Requirement |",
        "| --- | --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['status']} | `{check['key']}` | {check['evidence']} | "
            f"{check['requirement']} |"
        )
    lines.append("")
    if report["metadata"].get("compile_blocker"):
        lines.extend(
            [
                "## Compile Blocker",
                "",
                str(report["metadata"]["compile_blocker"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    checks, metadata = collect_checks()
    fail_count = sum(check.status == "fail" for check in checks)
    warn_count = sum(check.status == "warn" for check in checks)
    report = {
        "name": "VERA AAAI-27 source readiness audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_ready": fail_count == 0,
        "pdf_compile_ready": metadata.get("pdflatex") is not None,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": [asdict(check) for check in checks],
        "metadata": metadata,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("VERA AAAI-27 source readiness audit complete")
    print(f"source_ready={str(report['source_ready']).lower()}")
    print(f"pdf_compile_ready={str(report['pdf_compile_ready']).lower()}")
    print(f"fail_count={fail_count}")
    print(f"warn_count={warn_count}")
    print(f"report={args.json_out}")
    return 0 if args.no_fail or fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
