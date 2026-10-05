# Invoice Automator – Project Documentation

## 1. Project Title
Invoice Automator – Telegram-Based Invoice Extraction and Excel Reporting System

## 2. Student/Author
Mohammed Sharaf Saleh  
Internship Project – Implementing Course Knowledge in a Real-World Use Case

## 3. Problem Statement
Manual invoice processing is time-consuming, error-prone, and repetitive. Invoices arrive in different formats (digital PDFs with embedded text, scanned PDFs, or photos). Extracting key data (supplier, reference, dates, line items, totals, VAT) and compiling them into a clean, consistent Excel sheet requires significant manual effort and is prone to data entry errors.

## 4. Project Objective
To build an automated pipeline that:
- Receives invoices via Telegram (PDFs, images, photos).
- Automatically extracts structured invoice data (Arabic/English mixed).
- Validates totals and flags items needing review.
- Exports clean, properly formatted Excel (.xlsx) files with a fixed column layout.
- Runs automatically and requires minimal human intervention.

## 5. Solution
Developed a Telegram bot that batches incoming invoices, uses hybrid AI extraction (text-layer-first, vision fallback), performs math reconciliation, and produces a ready-to-use Excel report. The system is designed for reliability (deterministic validation), Arabic text rendering, and unattended operation.

## 6. Technologies & Tools Used
- **Python 3.12** – Core programming language
- **python-telegram-bot (v20+)** – Telegram Bot API integration (long-polling)
- **CrewAI** – Agent-based orchestration (extraction, decision, reporting)
- **OpenAI Python SDK** – Unified client for LLM calls
- **Groq API (qwen/qwen3.8-27b)** – Text and multimodal vision extraction (cost-efficient, low-latency)
- **PyMuPDF (fitz)** – PDF parsing, text extraction, image rendering
- **Pillow (PIL)** – Image preprocessing and tiling
- **openpyxl** – Excel (.xlsx) generation and styling
- **arabic-reshaper + python-bidi** – Arabic text normalization utilities
- **launchd (macOS)** – Persistent background service (user agent)
- **Docker** – Containerization for cloud deployment
- **Railway/Fly.io/Render configs** – Cloud deployment readiness

## 7. System Architecture
The pipeline follows a modular flow:
1. **Telegram Bot (bot.py)** – Receives files, debounces batches (3s), triggers extraction.
2. **CrewAI Orchestration (crews.py)** – Coordinates agents/tasks per stage.
3. **Agents (agents.py)** – Extraction, confirmation parsing, and report-building agents.
4. **Tasks (tasks.py)** – Defines agent instructions and outputs.
5. **Extraction/Tools (tools.py)** – Core hybrid extraction, math sanity checks, field validation, letterhead fallback.
6. **Excel Generator (file_generator.py)** – Builds denormalized workbook with exact column order and Arabic styling.
7. **Storage** – Temporary files written to `/tmp`, cleaned up after sending.

### Hybrid Extraction Strategy
- **Text-layer first (preferred):** For digital PDFs, extract embedded text via PyMuPDF. Send structured to LLM to normalize to schema (exact, no OCR).
- **Vision fallback:** If no text layer (scanned PDF/photo), render pages to images, tile into letterhead + body crops, send to Groq multimodal model.
- **Letterhead-only fallback:** If text layer exists but supplier name is only in a logo image, read top 15% strip with vision to recover supplier name without re-reading numbers.
- **Math reconciliation:** Validates qty*price == line_total and sum(line_totals) + VAT - discount == grand_total. Performs one corrective re-read on mismatch.
- **Readability checks:** Flags empty/unreadable critical fields (supplier/ref/date/method/lines) and sets `needs_review` with reason.

## 8. Data Schema (Output JSON)
```json
{
  "supplier_name": string,
  "buyer_name": string,
  "ref_number": string,
  "invoice_date": "YYYY-MM-DD",
  "payment_method": string,
  "payment_terms": string,
  "lines": [{"name": string, "quantity": number, "price_unit": number, "line_total": number}],
  "discount": number,
  "tax_amount": number,
  "grand_total": number,
  "needs_review": bool,
  "review_reason": string,
  "source_file": string,
  "extraction_mode": "text_layer" | "vision" | "failed"
}
```

## 9. Excel Output (Required Columns)
Exact order implemented:
1. `الرقم المرجعي` (ref_number)
2. `اسم المورد` (supplier_name)
3. `التاريخ` (invoice_date)
4. `الصنف` (item_name)
5. `المبلغ` (price_unit)
6. `الكمية` (quantity)
7. `مجموع المبلغ` (line_total)
8. `الخصم` (discount)
9. `الضريبة` (tax_amount)
10. `الإجمالي النهائي` (grand_total)
11. `طريقة الدفع` (payment_method)
12. `شروط الدفع` (payment_terms)
13. `يحتاج مراجعة` (needs_review – Arabic "نعم"/"لا")

Styling: Header blue background/white text; Arabic text rendered in Arial with RTL alignment (`readingOrder=2`); review rows highlighted; columns auto-sized.

## 10. How It Works (End-to-End)
1. User sends PDF/photo(s) to Telegram bot.
2. Bot batches files (debounce 3s) and shows extraction status.
3. Each file processed via hybrid extraction (text-layer → letterhead fallback → vision). Results normalized to schema.
4. Math check + readability validation applied; corrective pass if totals mismatch.
5. With `AUTO_SEND=true`, workbook built immediately and `.xlsx` sent back (no manual confirmation). Can be toggled to confirm-first.
6. Temporary files (inputs/outputs) deleted after sending. Session resets.

## 11. Key Features
- **Hybrid accuracy:** Prefers exact text layer; falls back to vision only when needed.
- **Arabic-first output:** Proper font, RTL, and verbatim transcription of Arabic text.
- **Data integrity:** Deterministic math reconciliation prevents silent arithmetic errors.
- **Robust to layout:** Handles logo-only supplier names via targeted letterhead crop.
- **Batching + debounce:** Groups multiple invoices in one Excel efficiently.
- **Auto-send mode:** Delivers results instantly for streamlined workflow.
- **Multi-user support:** Comma-separated allowed Telegram user IDs.
- **Persistent operation:** Runs as launchd user agent (macOS) or Docker container (cloud).

## 12. Deployment
- **Local (macOS):** `launchd` agent at `~/Library/LaunchAgents/com.invoiceautomator.bot.plist`, runs via `run_bot.sh`, auto-starts at login, keeps alive.
- **Container:** `Dockerfile` + `render.yaml` (Render Blueprint/Worker), `railway.json` (Railway), `fly.toml` (Fly.io). Configured as long-polling worker (no webhook required).

## 13. Configuration (.env)
Key variables:
- `TELEGRAM_BOT_TOKEN`, `ALLOWED_TELEGRAM_USER_ID` (comma-separated)
- `GROQ_API_KEY`
- `VISION_MODEL=qwen/qwen3.8-27b`, `TEXT_MODEL=qwen/qwen3.8-27b`, `DECISION_MODEL=openai/gpt-oss-120b`
- `TAX_RATE=0.15`, `AUTO_SEND=true`, `MAX_OUTPUT_TOKENS=900`
- Letterhead tuning: `LETTERHEAD_MAX_TOKENS`, `LETTERHEAD_MAX_ATTEMPTS`, `LETTERHEAD_RETRY_SECONDS`

## 14. Testing & Validation
- Text-layer PDF: exact Arabic extraction, correct math, no review.
- True scan (image-only PDF): vision fallback extracted supplier/ref/date/method/terms/lines/totals correctly.
- Logo-only supplier name case: letterhead vision pass recovered supplier without corrupting numeric fields.
- Column order, Arabic rendering, and review flagging verified against sample invoices.

## 15. Challenges & Solutions
- **Groq rate limits (429/413):** Capped output tokens, reduced vision input (2 tiles, 200 DPI, pixel cap), deduplicated prompts, added retries with backoff for letterhead pass.
- **Supplier name in logo image:** Implemented targeted top-band vision extraction to fill missing supplier field while preserving exact text-layer numbers.
- **Arabic rendering in Excel:** Enforced Arial font + RTL `readingOrder` for Arabic cells (default Calibri lacks Arabic glyphs).
- **Launchd path with spaces:** Used symlink (`/Users/user/invoice_bot`) to avoid space-related launch failures.
- **CrewAI tool schema edge case:** Parameterized report tool to emit valid JSON schema for Groq.

## 16. Learning Outcomes (Internship)
Applied course concepts in practice: API integration, async workflows, LLM tool use, data validation, file generation, process orchestration (CrewAI), containerization, and production-like service management (persistence, logging, cleanup). Gained experience with hybrid OCR/LLM extraction, multilingual (Arabic/English) data handling, and real-world error handling/rate-limiting.

## 17. Conclusion
The Invoice Automator successfully automates end-to-end invoice processing: from Telegram intake to clean, audit-ready Excel reports. The hybrid extraction balances accuracy (text-layer preferred) with resilience (vision fallback), while deterministic validation prevents silent errors. The solution meets the stated requirements, runs unattended, and is deployment-ready for local or cloud environments.
