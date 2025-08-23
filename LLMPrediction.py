from transformers import AutoTokenizer, AutoModelForCausalLM  # ✅ add this
import torch
import win32com.client
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_unread_emails():
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # 6 = Inbox
    messages = inbox.Items
    messages = messages.Restrict("[Unread] = true")
    
    emails = []
    for msg in messages:
        subject = msg.Subject or ""
        body = msg.Body or ""
        emails.append({
            "subject": subject.strip(),
            "body": body.strip()
        })
    return emails

emails = get_unread_emails()

labels = ["Need to look into", "Okay To be ignored", "Take a note"]

for email in emails:
    text = f"{email['subject']} {email['body']}"
    result = classifier(text, candidate_labels=labels)
    print(f"\n📩 Subject: {email['subject']}")
    print(f"Predicted: {result['labels'][0]} (score={result['scores'][0]:.2f})")
