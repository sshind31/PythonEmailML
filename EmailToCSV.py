import win32com.client
import pandas as pd

# Connect to Outlook
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder(6)  # 6 = Inbox

messages = inbox.Items
messages = messages.Restrict("[ReceivedTime] >= '01/01/2024'")  # filter (optional)

data = []

for mail in messages:
    try:
        if mail.Class == 43:  # 43 = MailItem
            subject = mail.Subject or ""
            body = mail.Body or ""
            sender = mail.SenderName or ""
            received = mail.ReceivedTime

            # Example rule: if you replied → label = 1 else 0
            label = 1 if mail.Categories and "Action Required" in mail.Categories else 0

            # Append row
            data.append({
                "subject": subject.replace("\n", " ").replace("\r", " "),
                "body": body.replace("\n", " ").replace("\r", " ")[:5000],  # truncate long mails
                "sender": sender,
                "received_time": received,
                "label": label
            })
    except Exception as e:
        print("Skipping item:", e)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("emails.csv", index=False, encoding="utf-8")
print("✅ Exported", len(df), "emails to emails.csv")
