from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import win32com.client

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

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
    prompt = f"Classify the following email into one of these categories: {labels}\n\nEmail:\n{text}\n\nAnswer with just the category name."
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n📩 Subject: {email['subject']}")
    print(f"Predicted: {prediction.strip()}")
