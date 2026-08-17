import os
import base64
import json
import time
import requests

from dotenv import load_dotenv
from groq import Groq
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SENT_IDS_FILE = "sent_ids.json"

client = Groq(api_key=GROQ_API_KEY)


# Send notifications through Telegram
def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": CHAT_ID,
                "text": text
            },
            timeout=20
        )

        print("Telegram response:", response.status_code)
        return response.status_code == 200

    except requests.exceptions.RequestException as e:
        print("Telegram error:", e)
        return False


# Connect to Gmail
def get_gmail_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


# Keep track of emails that have already been processed
def load_sent_ids():
    if os.path.exists(SENT_IDS_FILE):
        with open(SENT_IDS_FILE, "r") as file:
            return set(json.load(file))

    return set()


def save_sent_ids(ids):
    with open(SENT_IDS_FILE, "w") as file:
        json.dump(list(ids), file)


# Extract readable text from Gmail messages
def get_email_body(payload):
    body = ""

    if "parts" in payload:
        for part in payload["parts"]:

            if (
                part.get("mimeType") == "text/plain"
                and "data" in part.get("body", {})
            ):
                data = part["body"]["data"]

                body += base64.urlsafe_b64decode(
                    data
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

            elif "parts" in part:
                body += get_email_body(part)

    elif payload.get("body", {}).get("data"):
        data = payload["body"]["data"]

        body += base64.urlsafe_b64decode(
            data
        ).decode(
            "utf-8",
            errors="ignore"
        )

    return body.strip()


# Skip obvious promotional and social emails before using AI
def should_analyze_email(subject, sender):
    text = f"{subject} {sender}".lower()

    skip_keywords = [
        "unsubscribe",
        "newsletter",
        "weekly digest",
        "daily digest",
        "facebook",
        "instagram",
        "youtube",
        "amazon deals",
        "flipkart offers",
        "coupon",
        "discount",
        "cashback",
        "shopping",
        "food delivery",
        "swiggy",
        "zomato"
    ]

    for keyword in skip_keywords:
        if keyword in text:
            return False

    return True


# Analyze emails using Llama through Groq
def analyze_email(subject, sender, body):
    prompt = f"""
You are NeverMiss, an AI assistant that helps users avoid missing
important emails.

Analyze the email and return ONLY valid JSON.

Use exactly these fields:

{{
    "category": "INTERVIEW | ASSESSMENT | JOB | OFFER | REJECTION | DEADLINE | URGENT | PROMOTION | SOCIAL | NORMAL",
    "priority": "HIGH | MEDIUM | LOW",
    "company": "company name or Unknown",
    "role": "job role or Unknown",
    "summary": "maximum 25 words",
    "deadline": "date/time if mentioned, otherwise None",
    "action": "what the user should do, otherwise None",
    "application_link": "URL if present, otherwise None"
}}

Rules:

- Interview invitation → INTERVIEW
- Coding test, assessment or exam → ASSESSMENT
- Job opening or hiring notification → JOB
- Offer or selection confirmation → OFFER
- Rejection → REJECTION
- Important deadline → DEADLINE
- Time-sensitive important email → URGENT
- Marketing → PROMOTION
- Social media → SOCIAL
- Everything else → NORMAL

Priority:

- HIGH = interview, assessment, offer, urgent deadline or important action
- MEDIUM = job opportunity or useful information
- LOW = non-urgent information

Never invent information.
Use Unknown or None when information is missing.
Extract an application URL if one exists.
The user may apply for different types of jobs.
Do not calculate a job match score.
Keep the summary short.

From:
{sender}

Subject:
{subject}

Email:
{body[:4000]}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=400
        )

        result = response.choices[0].message.content.strip()

        print("Groq result:")
        print(result)

        return json.loads(result)

    except Exception as e:
        print("Groq error:", e)

        return {
            "category": "AI_ERROR",
            "priority": "LOW",
            "company": "Unknown",
            "role": "Unknown",
            "summary": "AI analysis failed. This email will be retried.",
            "deadline": "None",
            "action": "Retry later.",
            "application_link": "None"
        }


# Check unread Gmail messages and process new emails
def get_emails():
    service = get_gmail_service()

    print("Connected to Gmail!")

    sent_ids = load_sent_ids()

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q="is:unread",
        maxResults=10
    ).execute()

    messages = results.get("messages", [])

    print(f"Found {len(messages)} unread emails")

    new_messages = [
        message
        for message in messages
        if message["id"] not in sent_ids
    ]

    print(
        f"{len(new_messages)} are new "
        f"(not yet processed)"
    )

    if not new_messages:
        print("No new emails to process.")
        return

    for msg in new_messages:

        # Get the complete email
        email = service.users().messages().get(
            userId="me",
            id=msg["id"]
        ).execute()

        headers = email["payload"]["headers"]

        # Get the subject
        subject = next(
            (
                h["value"]
                for h in headers
                if h["name"].lower() == "subject"
            ),
            "No Subject"
        )

        # Get the sender
        sender_raw = next(
            (
                h["value"]
                for h in headers
                if h["name"].lower() == "from"
            ),
            "Unknown"
        )

        sender = sender_raw.split("<")[0].strip()

        # Skip obvious junk without spending Groq tokens
        if not should_analyze_email(subject, sender):
            print(f"Skipping without AI: {subject}")

            sent_ids.add(msg["id"])
            continue

        # Extract the email body
        body = get_email_body(email["payload"])

        print(f"\nAnalyzing: {subject}")

        # Ask Llama to analyze the email
        ai_result = analyze_email(
            subject,
            sender,
            body
        )

        # If AI fails, retry this email later
        if ai_result["category"] == "AI_ERROR":
            print(
                "AI analysis failed. "
                "Will retry later."
            )
            continue

        # Don't send unnecessary notifications
        if ai_result["category"] in [
            "NORMAL",
            "PROMOTION",
            "SOCIAL"
        ]:
            print(
                f"Skipping irrelevant email: "
                f"{subject} "
                f"[{ai_result['category']}]"
            )

            sent_ids.add(msg["id"])
            continue

        # Choose an icon based on priority
        priority = ai_result["priority"]

        if priority == "HIGH":
            icon = "🚨"
        elif priority == "MEDIUM":
            icon = "🔔"
        else:
            icon = "ℹ️"

        # Create the Telegram notification
        message = f"""
{icon} NEVERMISS — {ai_result["category"]}

🏢 Company: {ai_result["company"]}
💼 Role: {ai_result["role"]}

⚡ Priority: {priority}

📝 {ai_result["summary"]}

📅 Deadline: {ai_result["deadline"]}

👉 Action:
{ai_result["action"]}
"""

        if ai_result["application_link"] != "None":
            message += f"""
🔗 Application:
{ai_result["application_link"]}
"""

        message += f"""
━━━━━━━━━━━━━━━━━━━━
📧 From: {sender}
"""

        # Send the notification
        if send_telegram(message):
            sent_ids.add(msg["id"])

        time.sleep(3)

    # Save processed email IDs
    save_sent_ids(sent_ids)


# Start NeverMiss
if __name__ == "__main__":
    print("Starting NeverMiss...")
    get_emails()
    print("Done!")