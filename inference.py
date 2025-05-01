import torch
pip install transformers
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------- Load Model & Tokenizer --------------------
MODEL_PATH = "bert_fakenews_model.pth"  # Your saved .pth file
PRETRAINED_MODEL = "bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------- Prediction Function --------------------
def predict_fake_news(statement):
    encoding = tokenizer(statement, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    return pred, probs.cpu().numpy()
