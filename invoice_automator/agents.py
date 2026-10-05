"""CrewAI Agent definitions for the invoice pipeline.

Three agents, each mirroring one stage of the original pipeline:
  - extraction agent: reads an invoice file via its vision tool.
  - decision agent: interprets the owner's free-text confirmation reply.
  - report agent: builds the confirmed batch into an Excel file via its
    tool.

The extraction and report agents don't need a vision-capable brain
themselves - the real OCR/spreadsheet work happens inside their tools
(see tools.py). Their own LLM only ever has to decide to call that one
tool and relay its (verbatim, result_as_answer-locked) output, so a fast,
cheap text model is enough. The decision agent is the one place actual
language understanding is the point, so its LLM does the real reasoning.
"""

from __future__ import annotations

import os

from crewai import Agent, LLM
from dotenv import load_dotenv

from tools import extract_invoice

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DECISION_MODEL = os.getenv("DECISION_MODEL", "openai/gpt-oss-120b")


def groq_llm(model: str, **kwargs) -> LLM:
    """An LLM routed at Groq's OpenAI-compatible endpoint via crewai's
    native OpenAI client path - the same `openai` client our tools use
    directly, just wrapped for Agent use.

    Uses explicit provider="openai" rather than custom_openai=True:
    crewai's LLM factory (as of crewai 1.15.18) unconditionally strips a
    leading "openai/" from the model string whenever custom_openai=True is
    set, treating it as a redundant provider prefix. That's wrong for
    Groq's own catalog, where "openai/gpt-oss-120b" IS the literal model
    ID (Groq hosts OpenAI's open-weight gpt-oss models under that exact
    name) - stripping it sends the bare "gpt-oss-120b", which Groq
    rejects with model_not_found. Passing provider="openai" explicitly
    routes to the same native (non-litellm) OpenAI client class, honours
    our base_url/api_key exactly the same way, but leaves the model
    string untouched."""
    return LLM(
        model=model,
        provider="openai",
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=0,
        **kwargs,
    )


# Shared by the two tool-calling agents below - see module docstring.
_ORCHESTRATION_LLM = groq_llm(DECISION_MODEL)


def build_extraction_agent() -> Agent:
    return Agent(
        role="Invoice Extraction & Audit Specialist",
        goal=(
            "Read the supplier invoice at the given file path exactly as "
            "printed and return fully reconciled, review-flagged invoice "
            "data - never invent, round, or paraphrase a number or an "
            "Arabic word."
        ),
        backstory=(
            "A meticulous bookkeeper who never trusts a single glance at a "
            "receipt: you always call the extraction tool, which itself "
            "re-reads the page at multiple zoom levels, cross-checks the "
            "math, and corrects itself when the totals don't add up. You "
            "report exactly what it finds, without paraphrasing."
        ),
        tools=[extract_invoice],
        llm=_ORCHESTRATION_LLM,
        verbose=True,
    )


def build_decision_agent() -> Agent:
    return Agent(
        role="Confirmation Parser",
        goal=(
            "Turn the invoice batch owner's free-text Arabic or English "
            "reply into a precise, structured confirm/exclude decision, "
            "filling in only what was explicitly stated."
        ),
        backstory=(
            "A sharp bilingual assistant who reads casual replies like "
            "'أكد الكل ما عدا 3' or 'confirm only 1 and 2' and never "
            "assumes intent that wasn't said outright."
        ),
        llm=groq_llm(DECISION_MODEL),
        verbose=True,
    )


def build_report_agent(build_workbook_tool) -> Agent:
    return Agent(
        role="Report Builder",
        goal="Turn the confirmed invoices into a single clean Excel workbook.",
        backstory=(
            "A meticulous back-office clerk who assembles the final "
            "spreadsheet exactly the way accounting expects it, one row "
            "per line item, and hands back the saved file's path."
        ),
        tools=[build_workbook_tool],
        llm=_ORCHESTRATION_LLM,
        verbose=True,
    )
