"""Temporary output layer: builds Excel (.xlsx) from confirmed invoices.

This stands in for the future `stream_client.py` that will push to Odoo.
It is intentionally isolated: swapped out later without touching bot.py,
crews.py or tools.py.

One row per line item, with the invoice-level fields repeated on each of that
invoice's rows (denormalised).
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# Column keys in output order, as requested. The first eleven are the
# requested set; "شروط الدفع" and "يحتاج مراجعة" are kept after them for
# auditability (the review flag in particular is what marks rows that need a
# human eye, so dropping it would hide bad extractions).
COLUMNS = [
    ("ref_number", "الرقم المرجعي"),
    ("supplier_name", "اسم المورد"),
    ("invoice_date", "التاريخ"),
    ("item_name", "الصنف"),
    ("price_unit", "المبلغ"),
    ("quantity", "الكمية"),
    ("line_total", "مجموع المبلغ"),
    ("discount", "الخصم"),
    ("tax_amount", "الضريبة"),
    ("grand_total", "الإجمالي النهائي"),
    ("payment_method", "طريقة الدفع"),
    ("payment_terms", "شروط الدفع"),
    ("needs_review", "يحتاج مراجعة"),
]


def build_workbook(confirmed_invoices: list[dict]) -> Workbook:
    """Build an openpyxl Workbook from a list of confirmed invoice dicts.

    Args:
        confirmed_invoices: dicts as produced by ExtractionResult.to_dict(),
            already filtered to the confirmed subset.

    Returns:
        An openpyxl Workbook with one "Invoices" sheet.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Invoices"

    _write_header(ws)

    row_idx = 1
    for invoice in confirmed_invoices:
        lines = invoice.get("lines") or []
        if not lines:
            # An invoice with no parsed lines still gets one row so it is not
            # silently dropped.
            lines = [{}]

        for line in lines:
            row_idx += 1
            values = {
                "payment_method": _payment_method(invoice.get("payment_method", "")),
                "invoice_date": invoice.get("invoice_date", ""),
                "supplier_name": invoice.get("supplier_name", ""),
                "price_unit": _num(line.get("price_unit")),
                "item_name": line.get("name", ""),
                "quantity": _num(line.get("quantity")),
                "line_total": _num(line.get("line_total")),
                "discount": _num(invoice.get("discount")),
                "payment_terms": invoice.get("payment_terms", ""),
                "ref_number": invoice.get("ref_number", ""),
                "tax_amount": _num(invoice.get("tax_amount")),
                "grand_total": _num(invoice.get("grand_total")),
                "needs_review": "نعم" if invoice.get("needs_review") else "لا",
            }
            for col, (key, _label) in enumerate(COLUMNS, start=1):
                cell = ws.cell(row=row_idx, column=col, value=values[key])
                _style_data_cell(cell, values[key])

            _flag_review_row(ws, row_idx, invoice.get("needs_review"))

    _autosize(ws)
    ws.freeze_panes = "A2"
    return wb


def write_to_temp_file(confirmed_invoices: list[dict], label: str = "") -> str:
    """Persist the workbook to a temp path and return that path.

    The returned file is meant to be sent via Telegram and then deleted.

    `label` is an optional short batch label added to the filename; it is
    sanitized so a model-supplied string can never inject path separators.
    """
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "invoices"
    if label:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:40]
        if safe:
            prefix = f"invoices_{safe}"
    path = os.path.join(tempfile.gettempdir(), f"{prefix}_{suffix}.xlsx")
    wb = build_workbook(confirmed_invoices)
    wb.save(path)
    return path


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
# Calibri - the font openpyxl/Excel uses by default - ships no Arabic
# glyphs, so Arabic text renders as boxes or disconnected, unreadable
# letters even though the underlying characters are perfectly valid. Arial
# covers Arabic on both Windows and macOS, so we name it explicitly on every
# cell instead of relying on the default.
ARABIC_FONT = "Arial"


def _is_arabic(value) -> bool:
    return any(
        0x0600 <= ord(ch) <= 0x06FF or 0x0750 <= ord(ch) <= 0x077F
        for ch in str(value or "")
    )


def _write_header(ws) -> None:
    header_font = Font(name=ARABIC_FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill(
        start_color="4472C4", end_color="4472C4", fill_type="solid"
    )
    ws.append([label for _key, label in COLUMNS])
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _style_data_cell(cell, value) -> None:
    """Give a data cell an Arabic-capable font, and right-align + mark RTL
    when the content is Arabic so Excel displays it in the right order."""
    if isinstance(value, str) and _is_arabic(value):
        cell.font = Font(name=ARABIC_FONT)
        cell.alignment = Alignment(horizontal="right", readingOrder=2)
    elif isinstance(value, str):
        cell.font = Font(name=ARABIC_FONT)


def _flag_review_row(ws, row_idx: int, needs_review: bool) -> None:
    if not needs_review:
        return
    fill = PatternFill(
        start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"
    )
    for col in range(1, len(COLUMNS) + 1):
        ws.cell(row=row_idx, column=col).fill = fill


def _payment_method(method: str) -> str:
    mapping = {
        "نقدي": "نقدي (Cash)",
        "آجل": "آجل (Credit)",
        "مدى": "مدى (Mada)",
    }
    return mapping.get(method, method)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _autosize(ws) -> None:
    for col in range(1, len(COLUMNS) + 1):
        max_len = 0
        letter = get_column_letter(col)
        for cell in ws[letter]:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[letter].width = min(max_len + 4, 42)