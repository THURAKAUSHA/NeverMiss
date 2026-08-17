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


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]

SENT_IDS_FILE = "sent_ids.json"
FAIL_COUNTS_FILE = "fail_counts.json"
MAX_RETRIES = 2  # after this many failures, give up on the email and mark it processed

client = Groq(api_key=GROQ_API_KEY)


# SEND MESSAGE TO TELEGRAM

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


# CONNECT TO GMAIL

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


# REMEMBER PROCESSED EMAILS

def load_sent_ids():

    if os.path.exists(SENT_IDS_FILE):

        with open(SENT_IDS_FILE, "r") as file:

            return set(json.load(file))

    return set()


def save_sent_ids(ids):

    with open(SENT_IDS_FILE, "w") as file:

        json.dump(
            list(ids),
            file
        )


# TRACK PER-EMAIL AI FAILURE COUNTS (so broken emails don't retry forever)

def load_fail_counts():

    if os.path.exists(FAIL_COUNTS_FILE):

        with open(FAIL_COUNTS_FILE, "r") as file:

            return json.load(file)

    return {}


def save_fail_counts(counts):

    with open(FAIL_COUNTS_FILE, "w") as file:

        json.dump(
            counts,
            file
        )


# EXTRACT EMAIL BODY

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


# FILTER UNNECESSARY EMAILS

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


# ANALYZE EMAIL USING GROQ AI

def analyze_email(subject, sender, body):

    # Digest-style alerts ("X and 11 more new jobs") confuse the strict
    # single-job JSON schema below, since the model tries to describe
    # every listed job instead of one. Truncate the body harder for these
    # so the model only "sees" roughly the first job.
    if " and " in subject.lower() and "more" in subject.lower():
        body = body[:1500]
    else:
        body = body[:4000]

    prompt = f"""
You are NeverMiss, an AI career assistant.

Analyze this email for a job seeker.

Identify whether this email is:

INTERVIEW:
Interview invitation or interview scheduling.

ASSESSMENT:
Coding test, online assessment, hackathon, exam or test.

JOB:
Job opening, hiring notification or job alert.

OFFER:
Offer letter, selection confirmation or joining confirmation.

REJECTION:
Rejected, not selected or application unsuccessful.

DEADLINE:
Important application or assessment deadline.

URGENT:
Important time-sensitive email requiring attention.

PROMOTION:
Marketing, advertising, shopping or promotional email.

SOCIAL:
Social media notification.

NORMAL:
Everything else.

Priority rules:

HIGH:
Interview, assessment, offer, urgent issue or important action.

MEDIUM:
Useful job opportunity or useful career information.

LOW:
Non-urgent information.

Never invent information.

If this email lists multiple jobs, describe only the FIRST job mentioned
and summarize that the email also references other openings.

If information is missing:
company = Unknown
role = Unknown
deadline = None
action = None
application_link = None

Keep the summary short and simple.

Respond with ONLY a single JSON object with exactly these keys:
category, priority, company, role, summary, deadline, action, application_link.
No other text, no markdown, no code fences.

From:
{sender}

Subject:
{subject}

Email:
{body}
"""

    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "INTERVIEW",
                    "ASSESSMENT",
                    "JOB",
                    "OFFER",
                    "REJECTION",
                    "DEADLINE",
                    "URGENT",
                    "PROMOTION",
                    "SOCIAL",
                    "NORMAL"
                ]
            },
            "priority": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"]
            },
            "company": {"type": "string"},
            "role": {"type": "string"},
            "summary": {"type": "string"},
            "deadline": {"type": "string"},
            "action": {"type": "string"},
            "application_link": {"type": "string"}
        },
        "required": [
            "category", "priority", "company", "role",
            "summary", "deadline", "action", "application_link"
        ],
        "additionalProperties": False
    }

    SYSTEM_MESSAGE = (
        "You are NeverMiss AI. "
        "Analyze emails accurately and "
        "follow the required JSON structure."
    )

    def call_groq(response_format):
        return client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0.1,
            # gpt-oss-20b is a reasoning model: reasoning tokens count
            # against max_tokens, so keep this generous or the JSON can
            # get cut off mid-generation and fail schema validation.
            max_tokens=1200,
            reasoning_effort="low"
        )

    # ATTEMPT 1: strict json_schema mode (best guarantees, but has a
    # known Groq bug on gpt-oss-20b where it can 400 with an empty
    # failed_generation on some inputs)
    try:

        response = call_groq({
            "type": "json_schema",
            "json_schema": {
                "name": "never_miss_email",
                "strict": True,
                "schema": JSON_SCHEMA
            }
        })

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Groq returned an empty response.")

        result = result.strip()

        print("Groq result (schema mode):")
        print(result)

        return json.loads(result)

    except Exception as e:

        print("Groq schema-mode error, falling back to json_object mode:", e)

    # ATTEMPT 2: fallback to looser json_object mode. Less strict, but
    # community reports show it's far more reliable for gpt-oss-20b,
    # and our prompt already spells out the exact fields we want.
    try:

        response = call_groq({"type": "json_object"})

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Groq returned an empty response.")

        result = result.strip()

        print("Groq result (json_object fallback):")
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


# GET UNREAD EMAILS

def get_emails():

    service = get_gmail_service()

    print("Connected to Gmail!")

    sent_ids = load_sent_ids()
    fail_counts = load_fail_counts()

    results = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        q="is:unread",
        maxResults=10
    ).execute()

    messages = results.get(
        "messages",
        []
    )

    print(
        f"Found {len(messages)} unread emails"
    )

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


    # PROCESS EACH EMAIL

    for msg in new_messages:

        email = service.users().messages().get(
            userId="me",
            id=msg["id"]
        ).execute()

        headers = email["payload"]["headers"]


        # GET EMAIL SUBJECT

        subject = next(
            (
                h["value"]
                for h in headers
                if h["name"].lower() == "subject"
            ),
            "No Subject"
        )


        # GET EMAIL SENDER

        sender_raw = next(
            (
                h["value"]
                for h in headers
                if h["name"].lower() == "from"
            ),
            "Unknown"
        )

        sender = sender_raw.split("<")[0].strip()


        # SKIP OBVIOUS PROMOTIONAL EMAILS

        if not should_analyze_email(
            subject,
            sender
        ):

            print(
                f"Skipping without AI: {subject}"
            )

            sent_ids.add(msg["id"])

            continue


        # GET EMAIL BODY

        body = get_email_body(
            email["payload"]
        )


        # ANALYZE EMAIL

        print(
            f"\nAnalyzing: {subject}"
        )

        ai_result = analyze_email(
            subject,
            sender,
            body
        )


        # RETRY EMAIL IF AI FAILS, BUT GIVE UP AFTER MAX_RETRIES

        if ai_result["category"] == "AI_ERROR":

            fail_counts[msg["id"]] = fail_counts.get(msg["id"], 0) + 1

            if fail_counts[msg["id"]] >= MAX_RETRIES:

                print(
                    f"AI analysis failed {fail_counts[msg['id']]} times. "
                    f"Giving up on this email and marking it processed."
                )

                sent_ids.add(msg["id"])
                fail_counts.pop(msg["id"], None)

            else:

                print(
                    f"AI analysis failed "
                    f"({fail_counts[msg['id']]}/{MAX_RETRIES}). "
                    f"Will retry next run."
                )

            continue

        # Clear any prior failure count now that analysis succeeded
        fail_counts.pop(msg["id"], None)


        # SKIP UNIMPORTANT EMAILS

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


        # CHOOSE TELEGRAM ICON

        priority = ai_result["priority"]

        if priority == "HIGH":

            icon = "🚨"

        elif priority == "MEDIUM":

            icon = "🔔"

        else:

            icon = "ℹ️"


        # CREATE TELEGRAM MESSAGE

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


        # ADD APPLICATION LINK

        if ai_result["application_link"] != "None":

            message += f"""
🔗 Application:
{ai_result["application_link"]}
"""


        # ADD SENDER

        message += f"""
━━━━━━━━━━━━━━━━━━━━
📧 From: {sender}
"""


        # SEND TELEGRAM MESSAGE

        success = send_telegram(
            message
        )

        if success:

            sent_ids.add(
                msg["id"]
            )

        time.sleep(3)


    # SAVE PROCESSED EMAILS AND FAILURE COUNTS

    save_sent_ids(
        sent_ids
    )

    save_fail_counts(
        fail_counts
    )


# START NEVERMISS

if __name__ == "__main__":

    print(
        "Starting NeverMiss..."
    )

    get_emails()

    print(
        "Done!"
    )