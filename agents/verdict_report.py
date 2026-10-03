"""
Generate auditable PDF verdict scorecards from critic-agent reports.

Example:
    python agents/verdict_report.py AAPL --company "Apple Inc." --sector "Information Technology"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from agents.critic_agent import run_critic_agent
from agents.monitoring import log_verdict_run
from agents.policy_evidence import build_policy_evidence_trail


def _wrap(text: Any, width: int = 92) -> list[str]:
    if text is None:
        return []
    lines = []
    for raw_line in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(raw_line, width=width) or [""])
    return lines


def _draw_lines(ax, lines: list[str], x: float, y: float, size: int = 9, step: float = 0.028) -> float:
    for line in lines:
        if y < 0.06:
            break
        ax.text(x, y, line, fontsize=size, va="top", family="DejaVu Sans")
        y -= step
    return y


def _agent_score(output: dict[str, Any]) -> Any:
    return output.get("risk_score", output.get("macro_risk_score", "n/a"))


def _pdf_escape(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _plain_pdf_page(lines: list[str]) -> str:
    content = ["BT", "/F1 11 Tf", "50 770 Td", "14 TL"]
    for line in lines[:48]:
        content.append(f"({_pdf_escape(line)}) Tj")
        content.append("T*")
    content.append("ET")
    return "\n".join(content)


def _save_plain_pdf(report: dict[str, Any], output: Path) -> str:
    """Tiny dependency-free PDF writer used when matplotlib is unavailable."""
    final = report.get("final_output", {})
    lines = [
        f"Risk Verdict Scorecard: {report.get('ticker', 'UNKNOWN')}",
        f"Generated: {date.today().isoformat()}",
        "",
        f"Final Risk: {final.get('final_risk_label', final.get('risk_label', 'n/a'))} "
        f"({final.get('final_risk_score', final.get('risk_score', 'n/a'))}/10)",
        f"Confidence: {final.get('confidence', 'n/a')}",
        "",
        "Final decision:",
        *_wrap(final.get("final_decision", "No final decision captured."), width=76),
        "",
        "Agent scorecard:",
    ]
    for output_agent in report.get("agent_outputs", []):
        lines.extend(_wrap(
            f"- {output_agent.get('agent', 'Agent')}: "
            f"{output_agent.get('risk_label', 'n/a')} ({_agent_score(output_agent)}/10), "
            f"confidence={output_agent.get('confidence', 'n/a')}",
            width=76,
        ))

    lines.extend(["", "Policy A-H Evidence Trail:"])
    for item in final.get("policy_evidence_trail", []):
        lines.extend(_wrap(
            f"Policy {item.get('policy_id')} - {item.get('policy_name')} [{item.get('status')}]",
            width=76,
        ))
        for evidence in item.get("evidence", [])[:3]:
            lines.extend(_wrap(f"  - {evidence}", width=76))

    pages = [lines[i:i + 48] for i in range(0, len(lines), 48)] or [[]]
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
        f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(f'{3 + i * 2} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>\nendobj",
    ]
    for i, page_lines in enumerate(pages):
        page_obj = 3 + i * 2
        content_obj = page_obj + 1
        stream = _plain_pdf_page(page_lines)
        objects.append(
            f"{page_obj} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {content_obj} 0 R >>\nendobj"
        )
        objects.append(
            f"{content_obj} 0 obj\n<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\n"
            f"stream\n{stream}\nendstream\nendobj"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1", errors="replace")))
        pdf += obj + "\n"
    xref_offset = len(pdf.encode("latin-1", errors="replace"))
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    output.write_bytes(pdf.encode("latin-1", errors="replace"))
    return str(output)


def save_pdf_verdict(report: dict[str, Any], output_path: str | Path | None = None) -> str:
    """Save a multi-page PDF scorecard for a critic report."""
    final = report.get("final_output", {})
    if "policy_evidence_trail" not in final:
        final["policy_evidence_trail"] = build_policy_evidence_trail(report)

    ticker = report.get("ticker", "UNKNOWN")
    output = Path(output_path) if output_path else Path(config.LOCAL_VERDICTS) / f"{ticker}_risk_verdict.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return _save_plain_pdf(report, output)

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.96, f"Risk Verdict Scorecard: {ticker}", fontsize=19, weight="bold", va="top")
        ax.text(0.05, 0.925, f"Generated: {date.today().isoformat()}", fontsize=9, color="#555555", va="top")
        ax.text(
            0.05,
            0.88,
            f"Final Risk: {final.get('final_risk_label', final.get('risk_label', 'n/a'))} "
            f"({final.get('final_risk_score', final.get('risk_score', 'n/a'))}/10)",
            fontsize=14,
            weight="bold",
            va="top",
        )
        ax.text(0.05, 0.845, f"Confidence: {final.get('confidence', 'n/a')}", fontsize=11, va="top")
        y = 0.795
        y = _draw_lines(ax, ["Final decision:"], 0.05, y, size=11, step=0.032)
        y = _draw_lines(ax, _wrap(final.get("final_decision", "No final decision captured.")), 0.05, y, size=9)
        y -= 0.02
        y = _draw_lines(ax, ["Agent scorecard:"], 0.05, y, size=11, step=0.032)
        for output_agent in report.get("agent_outputs", []):
            line = (
                f"- {output_agent.get('agent', 'Agent')}: "
                f"{output_agent.get('risk_label', 'n/a')} ({_agent_score(output_agent)}/10), "
                f"confidence={output_agent.get('confidence', 'n/a')}"
            )
            y = _draw_lines(ax, _wrap(line), 0.07, y, size=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.96, "Policy A-H Evidence Trail", fontsize=17, weight="bold", va="top")
        y = 0.91
        for item in final.get("policy_evidence_trail", []):
            title = f"Policy {item.get('policy_id')} - {item.get('policy_name')} [{item.get('status')}]"
            y = _draw_lines(ax, [title], 0.05, y, size=11, step=0.032)
            for evidence in item.get("evidence", [])[:4]:
                y = _draw_lines(ax, _wrap(f"- {evidence}", width=88), 0.07, y, size=8.5, step=0.024)
            y -= 0.014
            if y < 0.12:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis("off")
                ax.text(0.05, 0.96, "Policy A-H Evidence Trail continued", fontsize=17, weight="bold", va="top")
                y = 0.91
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return str(output)


def save_json_verdict(report: dict[str, Any], output_path: str | Path | None = None) -> str:
    ticker = report.get("ticker", "UNKNOWN")
    output = Path(output_path) if output_path else Path(config.LOCAL_VERDICTS) / f"{ticker}_risk_verdict.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSON and PDF risk verdict scorecards.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL")
    parser.add_argument("--query", default="", help="Optional natural-language query for the critic.")
    parser.add_argument("--company", default=None, help="Optional company name.")
    parser.add_argument("--sector", default=None, help="Optional sector.")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM critic.")
    parser.add_argument("--json-out", default=None, help="Optional JSON output path.")
    parser.add_argument("--pdf-out", default=None, help="Optional PDF output path.")
    parser.add_argument("--log-monitoring", action="store_true", help="Log lineage to MLflow or local JSONL.")
    args = parser.parse_args()

    report = run_critic_agent(
        ticker=args.ticker,
        query=args.query,
        company_name=args.company,
        sector=args.sector,
        use_llm=args.use_llm,
    )
    json_path = save_json_verdict(report, args.json_out)
    pdf_path = save_pdf_verdict(report, args.pdf_out)
    if args.log_monitoring:
        tracking = log_verdict_run(report, run_type="critic_verdict", artifact_paths=[json_path, pdf_path])
        print(f"Logged monitoring via {tracking['tracking_backend']}")
    print(f"Saved JSON verdict: {json_path}")
    print(f"Saved PDF verdict:  {pdf_path}")


if __name__ == "__main__":
    main()
