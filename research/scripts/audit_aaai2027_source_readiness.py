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
SOURCE = AAAI_DIR / "vera_paper_body.tex"
ANONYMOUS_SOURCE = AAAI_DIR / "vera_aaai2027_anonymous.tex"
NAMED_SOURCE = AAAI_DIR / "vera_aaai2027_named.tex"
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
    body_source = load_text(SOURCE)
    anonymous_wrapper = load_text(ANONYMOUS_SOURCE)
    named_wrapper = load_text(NAMED_SOURCE)
    source = anonymous_wrapper + "\n" + body_source
    anonymous_source = anonymous_wrapper + "\n" + body_source
    named_source = named_wrapper + "\n" + body_source
    normalized_source = re.sub(r"\s+", " ", source)
    style = load_text(STYLE)
    texlive_pdflatex = Path("/Library/TeX/texbin/pdflatex")
    pdflatex = shutil.which("pdflatex") or (
        str(texlive_pdflatex) if texlive_pdflatex.is_file() else None
    )
    # The official AAAI-27 style aborts under Tectonic because it explicitly
    # requires pdfTeX. A generic TeX executable is therefore not enough.
    latex_engine = pdflatex
    word_count = estimate_main_words(body_source)
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
        "\\section{Related Work}",
        "\\section{Problem Setup}",
        "\\section{VERA}",
        "\\section{Experimental Protocol}",
        "\\section{Results}",
        "\\section{Reproducibility}",
        "\\section{Limitations and Conclusion}",
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
            status=status("\\usepackage[submission]{aaai2027}" in anonymous_wrapper),
            evidence="source uses \\usepackage[submission]{aaai2027}",
            requirement="Submission source must use the official AAAI submission style.",
        ),
        Check(
            key="anonymous_author_block",
            status=status(
                "\\author{Anonymous Submission}" in anonymous_wrapper
                and "VERA Project" not in anonymous_wrapper
                and "Rudra" not in anonymous_wrapper
                and "Chopra" not in anonymous_wrapper
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
                and "\\section{Reproducibility}" in anonymous_source
                and "anonymous supplement" in anonymous_source.lower()
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
                and "https://github.com/RudraChopra/mosaic-certified-release" in named_source
                and "\\section{Reproducibility}" in named_source
                and "\\newcommand{\\CodeAvailabilityText}" in named_source
            ),
            evidence=(
                f"path={NAMED_SOURCE}; materialized={materialized_file(NAMED_SOURCE)}; "
                f"has_author={'Rudra Chopra' in named_source}; "
                f"has_repo={'https://github.com/RudraChopra/mosaic-certified-release' in named_source}"
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
            key="official_baseline_matrix_present",
            status=status(
                all(term in source for term in ("INLP", "R-LACE", "LEACE", "TaCo", "MANCE++"))
                and "1,280 official-method" in source
                and "no proxy" in source.lower()
            ),
            evidence="source identifies all five official erasers, the current 1,280-run matrix, and the zero-proxy boundary",
            requirement="AAAI source must describe the current official baseline matrix without proxy rows.",
        ),
        Check(
            key="false_acceptance_corollary_present",
            status=status("false-acceptance control" in source.lower()),
            evidence="source contains the false-acceptance corollary",
            requirement="Theory section should include explicit false-acceptance control.",
        ),
        Check(
            key="reviewer_attack_preempted",
            status=status(
                "learn then test" in source.lower()
                and "does not claim" in source.lower()
                and "prompt risk control" in source.lower()
            ),
            evidence="source squarely attributes finite-family testing and distinguishes VERA from LTT and Prompt Risk Control",
            requirement="AAAI source should preempt the closest-prior-work objection.",
        ),
        Check(
            key="reference_boundary_present",
            status=status(
                all(term in source for term in ("R-LACE", "TaCo", "LEACE", "MANCE++"))
                and "no proxy" in source.lower()
                and "does not claim" in source.lower()
            ),
            evidence="source identifies pinned official baselines, excludes proxy rows, and states explicit non-novelty boundaries",
            requirement="AAAI source must not overclaim baseline or method novelty.",
        ),
        Check(
            key="clinical_boundary_present",
            status=status(
                "not clinical validation" in normalized_source
                and "clinical deployment evidence" in normalized_source
            ),
            evidence="source states Camelyon17/GaitPDB are not clinical deployment evidence",
            requirement="Medical benchmark language must not imply clinical deployment readiness.",
        ),
        Check(
            key="final_p0_negative_result_disclosed",
            status=status(
                "P0" in source
                and "negative confirmation" in source.lower()
                and "2 violations among 118 deployments" in source
            ),
            evidence="source discloses the final P0 non-superiority result against IID LTT",
            requirement="Current source must not let earlier follow-up results conceal the final P0 negative confirmation.",
        ),
        Check(
            key="estimated_length_reasonable",
            status=status(word_count <= 5600, warn=word_count <= 6500),
            evidence=f"estimated_main_words={word_count}",
            requirement="AAAI source should remain plausibly within the 7-page technical limit.",
        ),
        Check(
            key="latex_engine_available",
            status=status(latex_engine is not None, warn=True),
            evidence=f"latex_engine={latex_engine or '<missing>'}",
            requirement="Local final AAAI PDF compilation requires a pdfTeX-compatible PDFLaTeX executable.",
        ),
    ]
    metadata = {
        "source": str(SOURCE),
        "anonymous_source": str(ANONYMOUS_SOURCE),
        "named_source": str(NAMED_SOURCE),
        "style": str(STYLE),
        "estimated_main_words": word_count,
        "latex_engine": latex_engine,
        "compile_blocker": None if latex_engine else "PDFLaTeX is not installed locally; the AAAI style rejects Tectonic.",
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
        "pdf_compile_ready": metadata.get("latex_engine") is not None,
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
