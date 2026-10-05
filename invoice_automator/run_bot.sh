#!/bin/sh
# Wrapper so launchd can start the bot without needing a login shell or the
# user's interactive environment.
cd "/Users/user/Desktop/invoices automator/invoice_automator" || exit 1
exec "/Users/user/Desktop/invoices automator/invoice_automator/.venv/bin/python" bot.py