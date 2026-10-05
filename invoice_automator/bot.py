"""Telegram bot - async polling interface for the invoice pipeline.

Flow:
  1. Owner uploads photos/PDFs. A 3-second debounce collects them all into
     a single batch.
  2. Each file is extracted by the CrewAI extraction crew (vision + math
     sanity check, run concurrently across the batch).
  3. A numbered summary is sent back with "needs review" flags.
  4. The owner replies conversationally; the CrewAI decision crew parses it.
  5. Confirmed invoices are handed to the CrewAI report crew, which writes
     them to an .xlsx temp file; it's sent back as a document, then deleted.

The bot answers only the allowed Telegram user ID.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import crews

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _parse_allowed_user_ids(raw: str | None) -> set[int]:
    """Parse ALLOWED_TELEGRAM_USER_ID into a set of Telegram user IDs.

    Accepts a comma-separated list so more than one person can be allowed to
    use the bot. Entries that aren't valid integers are skipped with a
    warning rather than crashing the bot at import time, and 0 is never
    treated as a real ID.
    """
    ids: set[int] = set()
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            logger.warning(
                "Ignoring invalid entry %r in ALLOWED_TELEGRAM_USER_ID", part
            )
            continue
        if value != 0:
            ids.add(value)
    return ids


ALLOWED_USER_IDS = _parse_allowed_user_ids(os.getenv("ALLOWED_TELEGRAM_USER_ID"))

DEBOUNCE_SECONDS = 3

# When true, each batch is turned into an .xlsx and sent back as soon as the
# debounce window closes, with no confirmation step. Set AUTO_SEND=false to
# restore the "summarise, then wait for the owner to confirm" behaviour.
AUTO_SEND = os.getenv("AUTO_SEND", "true").strip().lower() in {
    "1", "true", "yes", "on",
}

# In-memory session. Single-owner bot, so one active session is enough.
_session: "_Session | None" = None


class _Session:
    """Tracks a pending batch waiting for confirmation."""

    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id
        self.files: list = []              # list of (file_path, original_name)
        self.invoices: list[dict] = []     # extracted dicts, aligned 1:1 with files
        self.debounce_task: asyncio.Task | None = None
        self.awaiting_confirmation = False


async def _restart_debounce(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Collect the current batch and process it after the debounce window."""
    global _session
    session = _session
    if session is None:
        return

    files = list(session.files)
    session.files = []

    if not files:
        return

    await context.bot.send_message(
        chat_id=session.chat_id,
        text=f"استلمت {len(files)} فاتورة. جاري الاستخراج...",
    )

    try:
        invoices = await crews.extract_invoices([file_path for file_path, _ in files])
        for (file_path, name), data in zip(files, invoices, strict=True):
            data["source_file"] = name
            logger.info("EXTRACTED %s -> %s", name,
                        json.dumps(data, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Extraction failed")
        await context.bot.send_message(
            chat_id=session.chat_id,
            text=f"حدث خطأ أثناء الاستخراج: {exc}",
        )
        return

    session.invoices = invoices

    if AUTO_SEND:
        await _send_file(context, session.chat_id, invoices)
        return

    session.awaiting_confirmation = True
    await _send_summary(context, session)


async def _send_file(context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                     invoices: list[dict]) -> None:
    """Build the workbook for the given invoices and send it to the chat.

    Shared by the auto-send path and the confirmation path so both produce
    and deliver the file identically.
    """
    flagged = [inv for inv in invoices if inv.get("needs_review")]
    summary = "\n".join(
        f"{i}. {inv.get('supplier_name') or 'غير معروف'} | "
        f"الإجمالي: {inv.get('grand_total', 0)} | "
        f"التاريخ: {inv.get('invoice_date') or '؟'}"
        f"{' ⚠️ يحتاج مراجعة' if inv.get('needs_review') else ''}"
        for i, inv in enumerate(invoices, start=1)
    )
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"📋 تم استخراج {len(invoices)} فاتورة:\n{summary}",
    )

    try:
        out_path = await crews.build_report(invoices)
        logger.info("Output file written: %s", out_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("File generation failed")
        await context.bot.send_message(chat_id=chat_id, text=f"فشل إنشاء الملف: {exc}")
        return

    try:
        await context.bot.send_document(
            chat_id=chat_id,
            document=out_path,
            filename=os.path.basename(out_path),
        )
        logger.info("Document sent to chat %s", chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sending document failed")
        await context.bot.send_message(chat_id=chat_id, text=f"فشل إرسال الملف: {exc}")
    finally:
        try:
            os.remove(out_path)
        except OSError:
            logger.warning("Could not delete temp file %s", out_path)

    if flagged:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ بعض الفواتير معلّمة «يحتاج مراجعة» في الملف - "
                 "يرجى التحقق منها.",
        )


async def _send_summary(context: ContextTypes.DEFAULT_TYPE,
                        session: _Session) -> None:
    lines = ["📋 ملخص الفواتير المستخرجة:\n"]
    for i, inv in enumerate(session.invoices, start=1):
        flag = " ⚠️ needs review" if inv.get("needs_review") else ""
        lines.append(
            f"{i}. {inv.get('supplier_name') or 'غير معروف'} | "
            f"الإجمالي: {inv.get('grand_total', 0)} | "
            f"التاريخ: {inv.get('invoice_date') or '؟'}{flag}"
        )
    lines.append("\nاكتب توجيهك لتأكيد الفواتير (مثال: 'أكد الكل' أو "
                 "'اعتمد فقط 1 و 2' أو 'أكد الكل ما عدا 3').")
    await context.bot.send_message(
        chat_id=session.chat_id, text="\n".join(lines)
    )


async def _handle_file(update: Update,
                       context: ContextTypes.DEFAULT_TYPE) -> None:
    global _session
    logger.info(
        "BEGIN _handle_file: user=%s allowed=%s photo=%s doc=%s mime=%s",
        update.effective_user.id,
        _is_allowed(update),
        bool(update.message.photo),
        bool(update.message.document),
        getattr(update.message.document, "mime_type", None),
    )
    if not _is_allowed(update):
        return
    if not _session:
        _session = _Session(chat_id=update.effective_chat.id)

    session = _session

    # Grab whichever media the message contains.
    item = (
        update.message.photo and update.message.photo[-1]
        or update.message.document
    )
    logger.info("Media handler: has_photo=%s has_document=%s mime=%s",
                bool(update.message.photo),
                bool(update.message.document),
                getattr(update.message.document, "mime_type", None))
    if item is None:
        await update.message.reply_text("أرسل صورة أو ملف PDF للفاتورة.")
        return

    file = await item.get_file()
    ext = ".pdf" if item.mime_type == "application/pdf" else ".jpg"
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    await file.download_to_drive(custom_path=temp_path)
    session.files.append((temp_path, item.file_name or "invoice"))

    # (Re)start the debounce timer.
    if session.debounce_task:
        session.debounce_task.cancel()
    session.debounce_task = asyncio.create_task(
        _debounce_and_process(context, session)
    )


async def _debounce_and_process(context: ContextTypes.DEFAULT_TYPE,
                                session: _Session) -> None:
    await asyncio.sleep(DEBOUNCE_SECONDS)
    try:
        await _restart_debounce(context)
    finally:
        session.debounce_task = None
        _cleanup_temp_files(session)


async def _handle_confirmation(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> None:
    global _session
    if not _is_allowed(update):
        return

    session = _session
    if not session or not session.awaiting_confirmation:
        if AUTO_SEND:
            await update.message.reply_text(
                "لا حاجة للتأكيد - سأرسل ملف الإكسل تلقائيًا بعد استلام "
                "الفواتير. أرسل صورة أو ملف PDF وسأنتظر ٣ ثوانٍ ثم أرسل الملف."
            )
        else:
            await update.message.reply_text(
                "لا توجد فواتير بانتظار التأكيد الآن."
            )
        return

    reply = update.message.text or ""
    logger.info("Confirmation received: %r over %d invoices", reply,
                len(session.invoices))
    decision = await crews.parse_confirmation(reply, num_invoices=len(session.invoices))
    logger.info("Decision: %s", decision)

    confirmed_indices = decision["confirmed_indices"]
    confirmed_invoices = [
        session.invoices[i - 1] for i in confirmed_indices
    ]
    logger.info("Confirmed %d invoice(s)", len(confirmed_invoices))

    if not confirmed_invoices:
        await update.message.reply_text("لم يتم تأكيد أي فاتورة. أُلغيت الدفعة.")
        _session = None
        return

    # Build the output file.
    try:
        out_path = await crews.build_report(confirmed_invoices)
        logger.info("Output file written: %s", out_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("File generation failed")
        await update.message.reply_text(f"فشل إنشاء الملف: {exc}")
        return

    msg = await update.message.reply_text(
        f"تم تأكيد {len(confirmed_invoices)} فاتورة. جاري إرسال الملف..."
    )
    try:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=out_path,
            filename=os.path.basename(out_path),
        )
        logger.info("Document sent to chat %s", update.effective_chat.id)
    finally:
        # Clean up the temp output file after sending.
        try:
            os.remove(out_path)
        except OSError:
            logger.warning("Could not delete temp file %s", out_path)

    _session = None


async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _session
    if not _is_allowed(update):
        return
    _session = None
    await update.message.reply_text(
        "أهلاً! أرسل لي صورًا أو ملفات PDF للفواتير. سأنتظر ٣ ثوانٍ بعد آخر "
        "ملف ثم أرسل لك ملف الإكسل مباشرة."
        if AUTO_SEND else
        "أهلاً! أرسل لي صورًا أو ملفات PDF للفواتير. سأنتظر قليلاً لتجميعها "
        "كدفعة واحدة ثم أعرض الملخص عليك للتأكيد."
    )


def _is_allowed(update: Update) -> bool:
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USER_IDS:
        logger.warning("Rejected access from user %s", user_id)
        return False
    return True


def _cleanup_temp_files(session: _Session) -> None:
    for file_path, _name in session.files:
        try:
            os.remove(file_path)
        except OSError:
            pass
    session.files = []


def main() -> None:
    if not (TOKEN and GROQ_API_KEY and ALLOWED_USER_IDS):
        raise SystemExit(
            "Missing TELEGRAM_BOT_TOKEN, GROQ_API_KEY or "
            "ALLOWED_TELEGRAM_USER_ID in environment."
        )

    logger.info("Allowed Telegram user IDs: %s", sorted(ALLOWED_USER_IDS))

    application = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", _start))

    # Images: Telegram photos (compressed) OR documents sent as files. Covers
    # both "send photo" and "send as file" modes, plus PDF documents.
    application.add_handler(
        MessageHandler(
            filters.PHOTO
            | filters.Document.PDF
            | filters.Document.IMAGE,
            _handle_file,
        )
    )
    # Any plain text while awaiting confirmation.
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_confirmation)
    )

    async def _log_errors(update, context):  # noqa: ANN001
        logger.error("Unhandled update error: %s", context.error, exc_info=True)

    application.add_error_handler(_log_errors)

    logger.info("Starting bot...")
    application.run_polling()


if __name__ == "__main__":
    main()
