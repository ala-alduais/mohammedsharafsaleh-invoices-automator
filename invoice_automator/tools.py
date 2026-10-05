"""CrewAI tools wrapping the precision-critical, deterministic parts of the
invoice pipeline: reading the invoice (PDF text layer first, Groq vision as
fallback), reconciling the totals, and building the output spreadsheet.

These stay as plain Python (called BY the Agents, not reasoned about BY
them) on purpose. A misread digit or a dropped Arabic word is not something
we want an LLM improvising over a second time - the Agents' job is to
decide *when* to call these tools and to relay their output, not to
re-derive or paraphrase the numbers themselves.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any

from crewai.tools import tool
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageFilter, ImageOps

import file_generator

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

# Loaded here (not just in bot.py) so this module's env-derived constants
# below are correct however it ends up being imported.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Multimodal FALLBACK extractor, used only when there's no text layer to read
# (scanned PDFs / plain images).
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
# Model used to structure a PDF's embedded TEXT layer - this is the preferred
# extraction path (exact text, no OCR, no vision involved).
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen/qwen3.8-27b")
TAX_RATE = float(os.getenv("TAX_RATE", "0.15"))

# Groq's on-demand service tier enforces a per-request output-token cap
# (~1000 OTPM), so we bound every completion explicitly rather than letting a
# model reason itself past the limit and get a 429.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "900"))

BUYER_NAME = "مؤسسة أقواس العارض"

# Vision-language models generally downsample a whole-page photo to some
# fixed internal resolution/token budget regardless of the source image's
# actual size. That means a single photo of a whole invoice can get
# internally downsampled enough that small Arabic script / multi-digit item
# codes are exactly what gets lost first - and the supplier name in the
# letterhead is often in a stylized/decorative font that's already hard to
# read even at full size. To work around this we tile a single page into a
# full overview, a dedicated zoomed crop of the letterhead (where the
# supplier name lives), and a zoomed crop of the rest of the page (customer
# info, items table, totals) - so each region gets more effective zoom than
# it would inside one generic top/bottom split. This helps regardless of
# which vision model is configured via VISION_MODEL.
# Groq's on-demand tier caps INPUT tokens per request (~7000 ITPM). A vision
# request costs roughly (width*height)/750 tokens per image, so these two
# knobs - a modest render DPI and a hard pixel-area cap per tile - are what
# keep a request inside that budget instead of failing outright with a 413.
PDF_RENDER_DPI = 200
MAX_IMAGE_PIXELS = 1_000_000
MAX_IMAGES_PER_REQUEST = 2
HEADER_HEIGHT_FRACTION = 0.28
BODY_START_FRACTION = 0.16
MIN_TILE_WIDTH = 1600
# Height of the top strip sent to the letterhead-only vision pass.
LETTERHEAD_BAND_FRACTION = 0.15
# The letterhead only needs a name back, so a small output budget plus a few
# retries rides out Groq's on-demand rate limits.
LETTERHEAD_MAX_TOKENS = int(os.getenv("LETTERHEAD_MAX_TOKENS", "120"))
LETTERHEAD_MAX_ATTEMPTS = int(os.getenv("LETTERHEAD_MAX_ATTEMPTS", "4"))
LETTERHEAD_RETRY_SECONDS = float(os.getenv("LETTERHEAD_RETRY_SECONDS", "8"))

logger = logging.getLogger(__name__)


_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ---------------------------------------------------------------------- #
# Image / PDF tiling helpers
# ---------------------------------------------------------------------- #
def _load_pages(file_path: str, is_pdf: bool) -> list[Image.Image]:
    """Return the source file as a list of full-resolution PIL Images."""
    if is_pdf:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required to read PDFs")
        pages = []
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append(img)
        doc.close()
        return pages
    return [Image.open(file_path).convert("RGB")]


def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    """Light preprocessing to help the model read the text: normalize
    contrast, sharpen slightly, and upscale small crops so text isn't any
    smaller than it has to be once the model downsamples to its fixed
    per-image token budget.

    Finally, hard-cap the pixel area - a request that exceeds Groq's
    input-token limit is rejected outright (413), so we bound each tile
    rather than relying on the model's internal downsampling."""
    img = ImageOps.exif_transpose(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    if img.width < MIN_TILE_WIDTH:
        scale = MIN_TILE_WIDTH / img.width
        img = img.resize(
            (MIN_TILE_WIDTH, int(img.height * scale)), Image.LANCZOS
        )

    pixels = img.width * img.height
    if pixels > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    return img


def _tile_page(page: Image.Image) -> list[tuple[str, Image.Image]]:
    """Split a single page into a dedicated zoomed crop of the
    letterhead/header (top of the page, where the supplier name lives) and a
    zoomed crop of everything below it. A generic top/bottom 50-50 split
    dilutes the header - which is usually only the top ~15-20% of the page -
    among a lot of blank space and the customer-info box, so the supplier
    name barely gets more effective zoom than in a full-page overview.
    Cropping the header on its own gives it roughly 3-4x the effective zoom
    instead.

    We deliberately send only these two crops and NOT a full-page overview:
    on Groq's on-demand tier the input-token budget is ~7000 tokens for the
    whole request, and a third image plus the prompt tips it over the limit
    (413) without adding information - the two crops already cover the whole
    page and overlap at the header/body boundary."""
    width, height = page.size
    header_crop = page.crop((0, 0, width, int(height * HEADER_HEIGHT_FRACTION)))
    body_crop = page.crop((0, int(height * BODY_START_FRACTION), width, height))

    return [
        ("Zoomed crop of the LETTERHEAD/HEADER at the TOP of this invoice "
         "page: supplier name, tax number, invoice number, date. The "
         "supplier name here is often a stylized logo font - read it letter "
         "by letter.",
         _enhance_for_ocr(header_crop)),
        ("Zoomed crop of the REST of the SAME invoice page: buyer info, "
         "items table and totals (it overlaps the header crop slightly). "
         "This is the same invoice as the header crop above, not a second "
         "invoice.",
         _enhance_for_ocr(body_crop)),
    ]


def _encode_png(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _prepare_vision_images(file_path: str, is_pdf: bool) -> list[tuple[str, str]]:
    """Load the source file and return [(caption, base64_png), ...] ready to
    hand to the vision model."""
    pages = _load_pages(file_path, is_pdf=is_pdf)
    if not pages:
        return []

    if len(pages) == 1:
        tiles = _tile_page(pages[0])
    else:
        # Multi-page PDF: tiling every page would blow past the model's
        # per-request image cap, so just enhance each page as-is.
        tiles = [
            (f"Page {i} of the invoice (of {len(pages)}).", _enhance_for_ocr(page))
            for i, page in enumerate(pages[:MAX_IMAGES_PER_REQUEST], start=1)
        ]

    return [(caption, _encode_png(img)) for caption, img in tiles]


# ---------------------------------------------------------------------- #
# Vision extraction (raw Groq call - deliberately NOT routed through a
# crewai Agent's own LLM; see module docstring)
# ---------------------------------------------------------------------- #
def _extra_body() -> dict:
    """Some vision models (e.g. Groq's qwen models) support a
    `reasoning_effort` knob ("none" | "default" | "low" | "medium" |
    "high"); pushing it to "high" makes those models deliberate more
    carefully over dense/small text instead of pattern-matching a fast (and,
    for Arabic script, error-prone) guess. Only send it when VISION_MODEL is
    actually a qwen model - non-Groq vendors don't support this
    parameter and would error (or silently ignore it) if it were sent."""
    if "qwen" in VISION_MODEL.lower():
        return {"reasoning_effort": "high"}
    return {}


def _build_image_content(vision_images: list[tuple[str, str]]) -> list:
    content: list = []
    for caption, b64 in vision_images:
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return content


_SCHEMA_BLOCK = (
    "Extract the following fields and return a JSON object with exactly "
    "these keys:\n"
    "{\n"
    '  "supplier_name": string,         // the VENDOR issuing the invoice\n'
    '  "ref_number": string,            // bill reference / invoice number\n'
    '  "invoice_date": "YYYY-MM-DD",    // invoice date\n'
    '  "payment_method": string,        // Arabic: نقدي / آجل / مدى / etc.\n'
    '  "payment_terms": string,         // credit/payment terms as printed\n'
    '                                   // e.g. "الدفع خلال 30 يوم", "نقدي عند\n'
    '                                   // الاستلام"; empty string if not stated\n'
    '  "lines": [                       // EVERY line item on the invoice\n'
    '    { "name": string, "quantity": number,\n'
    '      "price_unit": number, "line_total": number }\n'
    "  ],\n"
    '  "discount": number,              // total discount (0 if none)\n'
    '  "tax_amount": number,            // 15% VAT amount\n'
    '  "grand_total": number            // final total incl. tax\n'
    "}\n"
)

_SHARED_GUIDELINES = (
    "GUIDELINES:\n"
    f"- The buyer is ALWAYS '{BUYER_NAME}'. Never report the buyer as "
    "supplier_name; report only the SUPPLIER/vendor.\n"
    "- Transcribe every line item exactly; do not merge or skip rows.\n"
    "- Transcribe supplier_name, payment_terms and every line name VERBATIM, "
    "character for character, as printed. Do NOT 'correct', 'clean up', "
    "translate or 'improve' the spelling. Keep Arabic in Arabic and keep any "
    "product codes/SKUs on the line.\n"
    "- If a line has both a code/SKU and a description, keep them together "
    "separated by a space, exactly as printed.\n"
    "- Read digits carefully, one at a time; distinguish 0/8, 1/7, 3/8, 6/5, "
    "and do not transpose digits between an item code and a neighboring row's "
    "code.\n"
    "- Numbers may be in Western (0123) or Eastern Arabic (٠١٢٣) digits; "
    "convert them to Western digits in the output.\n"
    "- payment_terms is the credit/payment terms text as printed on the "
    "invoice (for example a stated credit period, an early-payment discount "
    "condition, or an explicit 'due on receipt' / cash note) - this is "
    "DIFFERENT from payment_method (نقدي / آجل / مدى). Leave it an empty "
    "string if the invoice does not print any such terms.\n"
    "- Report all numeric fields as plain numbers: no currency symbols, no "
    "thousand separators, no commas.\n"
    "- If any field is unreadable, use an empty string (text) or 0.0 (number); "
    "do NOT invent values.\n"
)


def _build_extraction_prompt() -> str:
    return (
        "You are a meticulous accountant reading a supplier invoice image. "
        "Extract the data EXACTLY as printed.\n"
        + _SCHEMA_BLOCK
        + _SHARED_GUIDELINES
        + "- You are given TWO zoomed crops of the SAME invoice page: a crop "
        "of the letterhead/header at the top, and a crop of the rest of the "
        "page. They are NOT two different invoices - cross-check them "
        "against each other and prefer whichever view lets you read a "
        "character with full confidence.\n"
        "- The supplier name in the letterhead is very often set in a "
        "stylized, calligraphic or decorative logo-style font that looks "
        "quite different from the plain print used elsewhere, which makes it "
        "easy to misread one or two letters and land on a different, "
        "unrelated-looking real-sounding name. Read it letter by letter, "
        "stroke by stroke; if a letter is genuinely ambiguous, prefer the "
        "reading that matches the letter shapes you can actually see over "
        "the one that merely 'sounds like a business name'.\n"
        "- Each row in the items table has EXACTLY ONE item code and ONE "
        "item name - never move or combine a code or name from one row into "
        "a different row. Before finalizing, recount the rows visible in "
        "the items table against the rows in your JSON.\n"
    )


def _build_text_prompt(correction: str = "") -> str:
    """Prompt for structuring a PDF's text layer (no image involved)."""
    prompt = (
        "You are a meticulous accountant. You are given the raw TEXT of a "
        "supplier invoice. This text was extracted directly from the PDF "
        "(not OCR), so it is reliable - read it carefully and transcribe the "
        "values EXACTLY as they appear.\n"
        + _SCHEMA_BLOCK
        + _SHARED_GUIDELINES
        + "- Keep the items-table rows in the same order as the source text. "
        "Each row has EXACTLY ONE item code and ONE item name - never move or "
        "combine a code or name from one row into a different row.\n"
        "- Watch out for text-extraction artifacts: column wrapping can split "
        "one row across several lines. Use the numeric columns (quantity, "
        "price, line total) to work out where each row actually starts and "
        "ends, and keep exactly one JSON entry per real table row.\n"
    )
    if correction:
        prompt += (
            "\nIMPORTANT: a previous attempt did not reconcile. The problem "
            f"was: {correction}\nRe-read the text carefully and return "
            "CORRECT numbers so that "
            "sum(line_total) + tax_amount - discount == grand_total.\n"
        )
    return prompt


def _parse_json_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        return {}


def _vision_extract(vision_images: list[tuple[str, str]]) -> dict:
    content = _build_image_content(vision_images)
    content.append({"type": "text", "text": _build_extraction_prompt()})

    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract invoice data from a photo of a supplier "
                    "invoice. Return ONLY valid JSON matching the requested "
                    "schema, with Arabic supplier names kept in Arabic. If "
                    "a value is unreadable, use an empty string or 0.0."
                ),
            },
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body=_extra_body(),
    )
    return _parse_json_response(response.choices[0].message.content)


def _vision_correct(vision_images: list[tuple[str, str]], result: dict) -> dict | None:
    """A second extraction pass focused on fixing math/number errors."""
    content = _build_image_content(vision_images)
    user = (
        _build_extraction_prompt() +
        "\n\nYour previous extraction did not reconcile. The numbers did "
        "not add up. Carefully re-read the invoice and return the CORRECT "
        "values. Pay close attention to the digits, especially quantities, "
        "prices and the totals. Recompute so that "
        "sum(line_total) + tax_amount - discount == grand_total."
    )
    content.append({"type": "text", "text": user})

    try:
        response = _client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are re-reading an Arabic supplier invoice to "
                        "fix number-extraction mistakes. Return ONLY valid "
                        "JSON with the corrected schema values. Read every "
                        "digit carefully."
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body=_extra_body(),
        )
        return _parse_json_response(response.choices[0].message.content)
    except Exception:  # noqa: BLE001
        return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------- #
# PDF text-layer extraction (preferred path - no OCR, no vision)
# ---------------------------------------------------------------------- #
def _pdf_to_text(file_path: str) -> str:
    """Return the PDF's embedded text layer.

    Returns an empty string for scanned/image-only PDFs (and for non-PDF
    inputs), which is the signal to fall back to the vision path.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to read PDFs")
    if not file_path.lower().endswith(".pdf"):
        return ""
    doc = fitz.open(file_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(pages).strip()


def _text_structure(raw_text: str, correction: str = "") -> dict:
    """Structure a PDF's text layer into invoice fields via a text model."""
    response = _client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meticulous accountant. You are given the raw "
                    "text of a supplier invoice and return ONLY a valid JSON "
                    "object of the requested schema. If a value is not "
                    "present in the text, use an empty string (text) or 0.0 "
                    "(number); never invent values."
                ),
            },
            {
                "role": "user",
                "content": f"{_build_text_prompt(correction)}\n\nINVOICE TEXT:\n{raw_text}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _parse_json_response(response.choices[0].message.content)


def _blank_result(file_path: str) -> dict:
    return {
        "supplier_name": "",
        "buyer_name": BUYER_NAME,
        "ref_number": "",
        "invoice_date": "",
        "payment_method": "",
        "payment_terms": "",
        "lines": [],
        "discount": 0.0,
        "tax_amount": 0.0,
        "grand_total": 0.0,
        "needs_review": False,
        "review_reason": "",
        "source_file": file_path,
        "extraction_mode": "",
    }


def _apply(result: dict, d: dict) -> None:
    if d.get("supplier_name"):
        result["supplier_name"] = d["supplier_name"]
    if d.get("ref_number"):
        result["ref_number"] = str(d["ref_number"])
    if d.get("invoice_date"):
        result["invoice_date"] = str(d["invoice_date"])
    if d.get("payment_method"):
        result["payment_method"] = str(d["payment_method"])
    if d.get("payment_terms"):
        result["payment_terms"] = str(d["payment_terms"])
    if d.get("lines"):
        result["lines"] = d["lines"]
    if d.get("discount") is not None:
        result["discount"] = _to_float(d["discount"])
    if d.get("tax_amount") is not None:
        result["tax_amount"] = _to_float(d["tax_amount"])
    if d.get("grand_total") is not None:
        result["grand_total"] = _to_float(d["grand_total"])


def _run_math_sanity_check(result: dict) -> None:
    """Compare computed total with the extracted grand_total, and verify
    each line item is internally consistent (qty * price == line_total)."""
    issues = []

    for idx, line in enumerate(result.get("lines") or [], start=1):
        qty = _to_float(line.get("quantity"))
        price = _to_float(line.get("price_unit"))
        reported = _to_float(line.get("line_total"))
        if reported:
            expected = qty * price
            tol = max(0.01, abs(expected) * 0.01)
            if abs(expected - reported) > tol:
                issues.append(
                    f"line {idx}: qty*price={expected:.2f} vs "
                    f"line_total={reported:.2f}"
                )

    lines = result.get("lines") or []
    lines_sum = sum(_to_float(line.get("line_total")) for line in lines)
    computed = lines_sum + result["tax_amount"] - result["discount"]
    extracted = result["grand_total"]

    tolerance = max(0.01, abs(computed) * 0.01)
    if abs(computed - extracted) > tolerance:
        issues.append(
            f"total: computed {computed:.2f} vs extracted {extracted:.2f}"
        )

    if issues:
        result["needs_review"] = True
        result["review_reason"] = "; ".join(issues)
    else:
        result["needs_review"] = False
        result["review_reason"] = ""


def _check_readable_fields(result: dict) -> None:
    """Flag invoices whose text could not actually be read.

    This runs AFTER _run_math_sanity_check and only ever adds to the review
    flag, so it can't interfere with the "total:" prefix that extract_invoice
    keys off to decide whether a corrective re-read is worthwhile.

    Two failure modes matter here and neither is caught by the math check:
    a completely empty extraction, and - more subtly - a PDF whose embedded
    text layer is itself corrupt (some generators emit '?' or U+FFFD for
    characters they cannot encode). Those can add up perfectly well, so the
    numbers reconcile and the invoice would otherwise sail through into the
    spreadsheet unmarked.
    """
    issues = []

    def _unreadable(value) -> bool:
        text = ("" if value is None else str(value)).strip()
        if not text:
            return True
        return "?" in text or "\ufffd" in text

    if not str(result.get("supplier_name") or "").strip():
        issues.append("supplier name is missing")
    elif _unreadable(result["supplier_name"]):
        issues.append(
            f"supplier name is unreadable: {result['supplier_name']!r}"
        )

    for key in ("ref_number", "invoice_date", "payment_method"):
        if _unreadable(result.get(key)):
            issues.append(f"{key} is unreadable")

    # payment_terms is OPTIONAL: many invoices state no credit period at all,
    # and the prompt already says to return an empty string in that case. An
    # empty value here is a legitimate answer, not a legibility failure, so
    # only flag it if it came back filled with unreadable characters.
    if _unreadable(result.get("payment_terms")) and str(
        result.get("payment_terms") or ""
    ).strip():
        issues.append("payment_terms is unreadable")

    lines = result.get("lines") or []
    if not lines:
        issues.append("no line items were read")
    else:
        for idx, line in enumerate(lines, start=1):
            if _unreadable(line.get("name")):
                issues.append(f"line {idx} name is unreadable")
            if _to_float(line.get("line_total")) == 0:
                issues.append(f"line {idx} has a zero line total")

    if not result["grand_total"]:
        issues.append("invoice total is zero")

    if issues:
        result["needs_review"] = True
        existing = result.get("review_reason") or ""
        result["review_reason"] = (
            f"{existing}; {issues}" if existing else "; ".join(issues)
        )


def _read_letterhead(file_path: str) -> str:
    """Read ONLY the top strip of the first page, where the supplier's logo /
    letterhead sits, and return just the supplier name it shows.

    Many real invoices are born-digital (so they have a perfect text layer for
    every number and line item) yet still print the supplier name ONLY inside
    a logo image across the top of the page. The text layer jumps straight
    from the document title to the buyer block, so the supplier name is
    simply absent from it. This targeted pass fills that one gap without
    re-reading - and risking - the numbers we already extracted exactly.
    """
    if fitz is None:
        return ""
    doc = fitz.open(file_path)
    try:
        page = doc[0]
        # The banner is a wide, short strip; take the top ~15% of the page.
        band = page.rect.height * LETTERHEAD_BAND_FRACTION
        clip = fitz.Rect(page.rect.x0, page.rect.y0, page.rect.x1, band)
        pix = page.get_pixmap(dpi=PDF_RENDER_DPI * 2, clip=clip)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()

    b64 = _encode_png(_enhance_for_ocr(img))
    prompt = (
        "This is the LETTERHEAD at the very top of a supplier invoice: the "
        "supplier's logo and company name. Read the SUPPLIER/VENDOR company "
        "name and return ONLY that name as a plain string - no JSON, no "
        "explanation, no extra text.\n"
        "The name is often set in a stylized or decorative logo font: read it "
        "letter by letter rather than guessing a name that merely sounds like "
        "a business.\n"
        "If no company name is visible, reply with exactly: NONE\n"
        f"The BUYER is always '{BUYER_NAME}' - never return the buyer as the "
        "supplier."
    )
    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    # Groq's on-demand tier is easy to trip (429 OTPM), and this pass only
    # needs a handful of tokens back, so retry a few times with backoff
    # before giving up. Giving up leaves the supplier name blank but flagged
    # for review - it never fails the whole invoice.
    response = None
    for attempt in range(LETTERHEAD_MAX_ATTEMPTS):
        try:
            response = _client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=LETTERHEAD_MAX_TOKENS,
                extra_body=_extra_body(),
            )
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Letterhead vision pass attempt %d/%d failed for %s: %s",
                attempt + 1, LETTERHEAD_MAX_ATTEMPTS, file_path, exc,
            )
            if attempt < LETTERHEAD_MAX_ATTEMPTS - 1:
                time.sleep(LETTERHEAD_RETRY_SECONDS * (attempt + 1))

    if response is None:
        return ""

    name = (response.choices[0].message.content or "").strip().strip('"')
    name = name.splitlines()[0].strip() if name else ""
    if not name or name.upper() == "NONE":
        return ""
    # Guard against the model handing back the buyer instead of the supplier.
    if BUYER_NAME and BUYER_NAME in name:
        return ""
    return name


@tool(result_as_answer=True)
def extract_invoice(file_path: str) -> str:
    """Read a supplier invoice from the given file path (PDF or image).

    For PDFs it first tries the document's embedded TEXT LAYER, which is
    exact (no OCR involved) and therefore the most accurate route. If there
    is no text layer - i.e. a scanned/image-only PDF or a plain image - it
    falls back to reading the image with a Groq vision model, using two
    zoomed crops of the page (letterhead, then the body).

    Either way it runs a deterministic math-reconciliation check, and if the
    totals don't add up it performs one corrective re-read pass. Finally it
    checks that the key fields were actually legible, so an unreadable or
    corrupt-text invoice is flagged rather than silently accepted.

    Returns a JSON string with keys: supplier_name, buyer_name, ref_number,
    invoice_date, payment_method, payment_terms, lines (list of {name,
    quantity, price_unit, line_total}), discount, tax_amount, grand_total,
    needs_review, review_reason, source_file, extraction_mode."""
    ext = os.path.splitext(file_path)[1].lower()
    is_pdf = ext == ".pdf"

    # ---- Preferred path: read the PDF's text layer (exact, no OCR) ----
    raw_text = _pdf_to_text(file_path) if is_pdf else ""
    if raw_text.strip():
        result = _blank_result(file_path)
        _apply(result, _text_structure(raw_text))
        _run_math_sanity_check(result)

        if result["needs_review"] and result["review_reason"].startswith("total:"):
            corrected = _text_structure(
                raw_text, correction=result["review_reason"]
            )
            if corrected:
                _apply(result, corrected)
                _run_math_sanity_check(result)

        _check_readable_fields(result)

        # The text layer is exact for the numbers, but it often does NOT
        # contain the supplier name at all - that lives in a logo image
        # across the top of the page. Fill just that one field from the
        # letterhead, leaving the already-exact values untouched.
        if not str(result.get("supplier_name") or "").strip() and is_pdf:
            letterhead = _read_letterhead(file_path)
            if letterhead:
                result["supplier_name"] = letterhead
                if "supplier name is missing" in (result["review_reason"] or ""):
                    result["review_reason"] = result["review_reason"].replace(
                        "supplier name is missing; ", ""
                    ).removesuffix("supplier name is missing")
                    if not result["review_reason"]:
                        result["needs_review"] = False

        result["extraction_mode"] = "text_layer"
        return json.dumps(result, ensure_ascii=False)

    # ---- Fallback: no text layer, so read the image with vision ----
    vision_images = _prepare_vision_images(file_path, is_pdf=is_pdf)
    if not vision_images:
        result = _blank_result(file_path)
        result["needs_review"] = True
        result["review_reason"] = "Could not read the file (no text, no image)"
        result["extraction_mode"] = "failed"
        return json.dumps(result, ensure_ascii=False)

    result = _blank_result(file_path)
    _apply(result, _vision_extract(vision_images))
    _run_math_sanity_check(result)

    # Corrective second pass: if the totals don't reconcile, ask the model
    # to re-read the numbers with the discrepancy in hand. This catches many
    # misread digit/line-item errors.
    if result["needs_review"] and result["review_reason"].startswith("total:"):
        corrected = _vision_correct(vision_images, result)
        if corrected:
            _apply(result, corrected)
            _run_math_sanity_check(result)

    _check_readable_fields(result)
    result["extraction_mode"] = "vision"
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------- #
# Report building
# ---------------------------------------------------------------------- #
def make_build_workbook_tool(confirmed_invoices: list[dict]):
    """Build a fresh, single-use tool with the confirmed invoices already
    baked in via closure.

    The invoice data (numbers, Arabic names) never passes back through an
    LLM's own token generation this way - the agent only decides *to* call
    the tool, it never has to copy the data into the call itself, so there's
    no chance of a digit or a word drifting on the way through.
    """

    @tool("build_invoices_workbook", result_as_answer=True)
    def build_invoices_workbook(label: str = "") -> str:
        """Build the confirmed invoices (already attached - no invoice data
        needs to be passed in) into a single Excel (.xlsx) workbook, one row
        per line item, and save it to a temp file. Returns the absolute path
        to the saved file.

        `label` is optional: a short batch label for the filename. Leave it
        empty to use the default name.
        """
        return file_generator.write_to_temp_file(confirmed_invoices, label=label)

    return build_invoices_workbook
