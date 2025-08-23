import tkinter as tk
import win32com.client
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

root = tk.Tk()
root.title("Overlay")
root.geometry("300x100+1500+100")  # width x height + x + y

# Remove window border
root.overrideredirect(True)

# Always on top
root.wm_attributes("-topmost", True)

# Semi-transparent
root.wm_attributes("-alpha", 0.5)

# Frame for buttons
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=5)

# Refresh button updates label text
btn_refresh = tk.Button(
    button_frame,
    text="🔄 Refresh",
    command=lambda: label.config(text=LoadUnreadMails())
)
btn_refresh.pack(side="left", padx=5)

# Close button
btn_close = tk.Button(button_frame, text="❌ Close", command=root.destroy)
btn_close.pack(side="left", padx=5)

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

text=""
def LoadUnreadMails():
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
    output_text = ""   # accumulator for overlay text

    for mail in unread_mails:
        mail_text = f"{mail['subject']} {mail['body']}"  # <- local var for classification
        result = classifier(mail_text[:512])[0]  # truncate long emails to 512 tokens
        label = result["label"]
        score = result["score"]

        if label == "IMPORTANT" and score > 0.8:
            output_text += "\n🚨 Important: " + mail['subject'] + " (score " + str(round(score, 2)) + ")"
            print(f"🚨 Important: {mail['subject']} (score {score:.2f})")
        else:
            output_text += "\n✅ Not Important: " + mail['subject'] + " (score " + str(round(score, 2)) + ")"
            print(f"✅ Not Important: {mail['subject']} (score {score:.2f})")

    return output_text if output_text else "📭 No unread mails"


# label = tk.Label(root, text="🔔 Overlay Example", font=("Arial", 16), bg="yellow")
label = tk.Label(root, text=LoadUnreadMails(), font=("Arial", 8), bg="white", wraplength=300, justify="left")
label.pack(expand=True, fill="both")

root.mainloop()
