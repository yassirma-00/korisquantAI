"""Transactional email for verification and password reset.

Degrading honestly without SMTP
-------------------------------
With no ``SMTP_HOST`` configured the message is not silently dropped: it is
written to the server log and the link is handed back to the caller, which the
UI then shows on screen. The whole flow — token, expiry, single-use purpose —
is exactly the production one; only the transport changes. Pretending an email
was sent when nothing left the machine is the failure mode this avoids, because
the user then waits forever for a message that never existed.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _shell(title: str, intro: str, button_label: str, link: str,
           footer: str) -> str:
    """A self-contained HTML email.

    Inline styles and a table layout on purpose: email clients strip <style>
    blocks and have no CSS grid. The plain-text alternative always carries the
    raw URL so the message stays usable when images and HTML are blocked.
    """
    return f"""\
<!DOCTYPE html><html><body style="margin:0;padding:24px;background:#f6f7fc;
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table role="presentation" width="520" cellpadding="0" cellspacing="0"
 style="background:#ffffff;border-radius:14px;overflow:hidden;
 box-shadow:0 2px 8px rgba(19,26,46,.07)">
<tr><td style="padding:26px 30px 0">
  <div style="font-size:19px;font-weight:800;color:#131a2e;letter-spacing:-.3px">
    <span style="display:inline-block;width:30px;height:30px;line-height:30px;
     text-align:center;border-radius:9px;background:#5a48e0;color:#fff;
     font-size:16px;margin-right:9px">K</span>KorisQuant AI</div>
</td></tr>
<tr><td style="padding:22px 30px 0">
  <h1 style="margin:0 0 12px;font-size:20px;color:#131a2e">{title}</h1>
  <p style="margin:0 0 20px;font-size:14px;line-height:1.6;color:#3d4861">{intro}</p>
  <a href="{link}" style="display:inline-block;background:#5a48e0;color:#ffffff;
   text-decoration:none;padding:12px 22px;border-radius:10px;font-size:14px;
   font-weight:600">{button_label}</a>
  <p style="margin:20px 0 0;font-size:12px;line-height:1.6;color:#5d6982">
    Or paste this link into your browser:<br>
    <span style="word-break:break-all;color:#5a48e0">{link}</span></p>
</td></tr>
<tr><td style="padding:22px 30px 26px">
  <hr style="border:none;border-top:1px solid #e6eaf5;margin:0 0 14px">
  <p style="margin:0;font-size:11.5px;line-height:1.6;color:#5d6982">{footer}</p>
</td></tr></table>
<p style="margin:16px 0 0;font-size:11px;color:#5d6982">
  © KorisQuant AI · Educational and research software, not investment advice</p>
</td></tr></table></body></html>"""


def _send(to: str, subject: str, html: str, text: str) -> bool:
    """Deliver over SMTP. Returns False when there is nothing configured."""
    if not settings.email_enabled:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.SMTP_FROM
    message["To"] = to
    message.set_content(text)
    message.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as smtp:
            if settings.SMTP_USE_TLS:
                smtp.starttls()
            if settings.SMTP_USER:
                smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD or "")
            smtp.send_message(message)
        logger.info("sent %r to %s", subject, to)
        return True
    except Exception as exc:
        # A delivery failure must not fail the request that triggered it: the
        # account still exists and the user can ask for another link.
        logger.warning("could not email %s: %s", to, exc)
        return False


def send_verification(to: str, username: str, token: str) -> dict:
    link = f"{settings.PUBLIC_BASE_URL}/auth.html?verify={token}"
    html = _shell(
        "Confirm your email",
        f"Welcome, {username}. Confirm this address to activate your "
        "KorisQuant AI account.",
        "Confirm my email", link,
        "This link expires in 24 hours. If you did not create an account, "
        "you can ignore this message.")
    text = (f"Welcome, {username}.\n\nConfirm your email to activate your "
            f"KorisQuant AI account:\n{link}\n\nThis link expires in 24 hours.")
    delivered = _send(to, "Confirm your KorisQuant AI account", html, text)

    if not delivered:
        # Loud enough to find in a busy log: without SMTP this line *is* the
        # delivery mechanism.
        logger.info("EMAIL NOT CONFIGURED - verification link for %s: %s", to, link)
    return {"delivered": delivered, "link": None if delivered else link}
