"""Public entry points bot.py calls into: one small Crew per pipeline stage.

Each stage is its own Crew (rather than one Crew for the whole pipeline)
because the Telegram flow isn't linear in a single request/response - it
spans separate incoming messages (files arrive, then later a confirmation
reply, then later still nothing until the next batch) - so each stage is
kicked off independently, exactly when that message arrives.
"""

from __future__ import annotations

import json
import logging

from crewai import Crew, Process

from agents import build_decision_agent, build_extraction_agent, build_report_agent
from tasks import build_decision_task, build_extraction_task, build_report_task
from tools import make_build_workbook_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Extraction
# ---------------------------------------------------------------------- #
async def extract_invoices(file_paths: list[str]) -> list[dict]:
    """Run the extraction crew once per file, concurrently, preserving
    input order so callers can keep matching results back to files by
    index (the Telegram confirmation flow refers to invoices as "1", "2",
    ...)."""
    if not file_paths:
        return []

    agent = build_extraction_agent()
    task = build_extraction_task(agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    outputs = await crew.kickoff_for_each_async(
        inputs=[{"file_path": path} for path in file_paths]
    )

    invoices = []
    for path, output in zip(file_paths, outputs, strict=True):
        try:
            invoices.append(json.loads(output.raw))
        except json.JSONDecodeError:
            logger.error("Extraction agent returned non-JSON for %s: %s",
                        path, output.raw)
            invoices.append({
                "supplier_name": "", "buyer_name": "", "ref_number": "",
                "invoice_date": "", "payment_method": "", "payment_terms": "",
                "lines": [],
                "discount": 0.0, "tax_amount": 0.0, "grand_total": 0.0,
                "needs_review": True,
                "review_reason": "extraction agent returned invalid JSON",
                "source_file": path,
            })
    return invoices


# ---------------------------------------------------------------------- #
# Confirmation parsing
# ---------------------------------------------------------------------- #
def _clean_indices(values: list[int] | None, num_invoices: int) -> list[int]:
    if not values:
        return []
    return sorted({i for i in values if 1 <= i <= num_invoices})


def _resolve(confirm_all: bool, confirmed: list[int], excluded: list[int],
            num_invoices: int) -> dict:
    """Combine the agent's interpretation into a deterministic decision -
    kept as plain code (not a tool/agent) so we stay robust even if the
    model is loose with edge cases like overlapping confirm/exclude lists."""
    all_indices = list(range(1, num_invoices + 1))

    confirmed = _clean_indices(confirmed, num_invoices)
    excluded = _clean_indices(excluded, num_invoices)

    if confirm_all:
        confirmed = [i for i in all_indices if i not in excluded]
    elif not excluded:
        # Only specific confirmations given (e.g. "only the first two").
        pass
    else:
        # Some exclusions given without confirm_all -> approve the rest.
        confirmed = [i for i in confirmed if i not in excluded]

    return {
        "confirm_all": confirm_all,
        "confirmed_indices": sorted(set(confirmed)),
        "excluded_indices": sorted(set(excluded)),
    }


async def parse_confirmation(reply: str, num_invoices: int) -> dict:
    """Parse the owner's free-text reply into
    {confirm_all, confirmed_indices, excluded_indices}."""
    if not reply or not reply.strip():
        return {"confirm_all": False, "confirmed_indices": [], "excluded_indices": []}

    agent = build_decision_agent()
    task = build_decision_task(agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    output = await crew.kickoff_async(
        inputs={"reply": reply, "num_invoices": num_invoices}
    )

    decision = output.pydantic
    if decision is None:
        logger.error("Decision agent did not return structured output: %s", output.raw)
        return {"confirm_all": False, "confirmed_indices": [], "excluded_indices": []}

    return _resolve(decision.confirm_all, decision.confirmed, decision.excluded,
                    num_invoices)


# ---------------------------------------------------------------------- #
# Report building
# ---------------------------------------------------------------------- #
async def build_report(confirmed_invoices: list[dict]) -> str:
    """Build the confirmed invoices into an .xlsx file and return its path."""
    build_workbook_tool = make_build_workbook_tool(confirmed_invoices)
    agent = build_report_agent(build_workbook_tool)
    task = build_report_task(agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    output = await crew.kickoff_async()
    return output.raw.strip()
