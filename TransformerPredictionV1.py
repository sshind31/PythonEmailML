import win32com.client
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

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


model_path = "./email_model_transformer"  # your trained model dir

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

unread_mails = get_unread_emails()

for mail in unread_mails:
    text = f"{mail['subject']} {mail['body']}"
    result = classifier(text[:512])[0]  # truncate long emails to 512 tokens
    label = result["label"]
    score = result["score"]

    if label == "IMPORTANT" and score > 0.8:
        print(f"🚨 Important: {mail['subject']} (score {score:.2f})")
    else:
        print(f"✅ Not Important: {mail['subject']} (score {score:.2f})")
