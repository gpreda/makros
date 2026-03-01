"""Email notifications for makros (best-effort, non-blocking)."""

import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SMTP_EMAIL = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASS")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


def send_email(to_email: str, subject: str, html_body: str):
    """Send an email via Gmail SMTP. Raises on failure."""
    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())


def notify_goal_added(client_email: str, client_name: str, coach_name: str,
                      goal_text: str, date: str):
    """Send goal-added notification in a background thread. Never raises."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return

    subject = f"New goal from your coach {coach_name}"
    html_body = f"""\
<html><body>
<p>Hi {client_name or "there"},</p>
<p>Your coach <strong>{coach_name}</strong> set a goal for you on <strong>{date}</strong>:</p>
<blockquote style="border-left:3px solid #4CAF50;padding:8px 12px;margin:8px 0;background:#f9f9f9;">
{goal_text}
</blockquote>
<p>Open <a href="https://makros.pr3da.com">Makros</a> to view it.</p>
</body></html>"""

    def _send():
        try:
            send_email(client_email, subject, html_body)
        except Exception as e:
            print(f"[notifications] Failed to send goal email to {client_email}: {e}")

    threading.Thread(target=_send, daemon=True).start()


def notify_goal_completed(coach_email: str, coach_name: str, client_name: str,
                          goal_text: str):
    """Notify a coach that their client completed a goal. Never raises."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return

    subject = f"{client_name or 'Your client'} completed a goal!"
    html_body = f"""\
<html><body>
<p>Hi {coach_name or "there"},</p>
<p>Your client <strong>{client_name}</strong> just completed a goal:</p>
<blockquote style="border-left:3px solid #4CAF50;padding:8px 12px;margin:8px 0;background:#f9f9f9;">
{goal_text}
</blockquote>
<p>Open <a href="https://makros.pr3da.com">Makros</a> to check their progress.</p>
</body></html>"""

    def _send():
        try:
            send_email(coach_email, subject, html_body)
        except Exception as e:
            print(f"[notifications] Failed to send completion email to {coach_email}: {e}")

    threading.Thread(target=_send, daemon=True).start()
