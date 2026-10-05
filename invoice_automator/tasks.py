"""CrewAI Task definitions for the invoice pipeline."""

from __future__ import annotations

from crewai import Task
from pydantic import BaseModel, Field


class ConfirmationDecision(BaseModel):
    """Structured shape the Confirmation Parser agent must return."""

    confirm_all: bool = Field(
        description="True if the owner approved every invoice in the batch."
    )
    confirmed: list[int] = Field(
        default_factory=list,
        description="1-based indices explicitly approved by the owner.",
    )
    excluded: list[int] = Field(
        default_factory=list,
        description="1-based indices explicitly excluded/rejected by the owner.",
    )


def build_extraction_task(agent) -> Task:
    """{file_path} is filled in per invoice via kickoff_for_each_async
    inputs - a short, opaque path string is safe for the agent's LLM to
    copy into its tool call; the actual invoice content never has to pass
    through the agent's own text generation."""
    return Task(
        description=(
            "Extract the invoice data from the file at this exact path: "
            "{file_path}\n\n"
            "Call your invoice extraction tool ONCE with that file path. "
            "The tool already renders the page at multiple zoom levels, "
            "calls the vision model, reconciles the totals, and corrects "
            "itself if needed - do not re-derive, reformat, round, "
            "translate, or otherwise edit any value it returns. Your final "
            "answer must be exactly the JSON string the tool returned."
        ),
        expected_output=(
            "The exact JSON object returned by the extraction tool, with "
            "keys supplier_name, buyer_name, ref_number, invoice_date, "
            "payment_method, lines, discount, tax_amount, grand_total, "
            "needs_review, review_reason, source_file."
        ),
        agent=agent,
    )


def build_decision_task(agent) -> Task:
    """{reply} and {num_invoices} are filled in via kickoff inputs."""
    return Task(
        description=(
            "There are {num_invoices} invoice(s) in the current batch, "
            "numbered 1..{num_invoices}. The batch owner replied with this "
            "free-text instruction (Arabic or English), given here between "
            "triple quotes:\n"
            '"""{reply}"""\n\n'
            "Interpret it into a confirm/exclude decision:\n"
            "- If the owner approved every invoice (e.g. 'أكد الكل', "
            "'confirm all', 'موافق على الكل'), set confirm_all to true and "
            "leave confirmed/excluded empty.\n"
            "- If the owner named specific invoices to exclude while "
            "otherwise approving the rest (e.g. 'أكد الكل ما عدا 3', "
            "'confirm all except 2'), set confirm_all to true and put the "
            "excluded 1-based indices in excluded.\n"
            "- If the owner named only specific invoices to approve (e.g. "
            "'اعتمد فقط 1 و 2', 'only confirm 1 and 2'), set confirm_all to "
            "false and put those indices in confirmed.\n"
            "- Fill in only what the owner explicitly stated - never guess "
            "at indices they didn't mention."
        ),
        expected_output=(
            "A structured decision with confirm_all, confirmed (1-based "
            "indices), and excluded (1-based indices)."
        ),
        output_pydantic=ConfirmationDecision,
        agent=agent,
    )


def build_report_task(agent) -> Task:
    """No placeholders - the confirmed invoice data is bound directly into
    the agent's tool via closure (see tools.make_build_workbook_tool), not
    threaded through this description, so it never has to pass through the
    agent's own text generation."""
    return Task(
        description=(
            "Call your build-invoices-workbook tool (it takes no "
            "arguments - the confirmed invoice data is already attached) "
            "to generate the Excel file for this confirmed batch. Your "
            "final answer must be exactly the file path the tool returned, "
            "nothing else."
        ),
        expected_output="The absolute file path returned by the tool, and nothing else.",
        agent=agent,
    )
