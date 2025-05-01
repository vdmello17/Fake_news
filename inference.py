import torch
import torch.nn as nn
import re
from collections import Counter
import numpy as np
import os
import gdown

# -------------------- Preprocessing --------------------
def tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()

sample_texts = ["the president said this is not true", "cnn reported fake news"]
tokenized = [tokenize(text) for text in sample_texts]
counter = Counter(token for tokens in tokenized for token in tokens)

vocab = {"<pad>": 0, "<unk>": 1}
vocab.update({word: i + 2 for i, (word, _) in enumerate(counter.items())})

def encode(tokens, max_len=100):
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    padded = ids[:max_len] + [0] * (max_len - len(ids))
    return padded

# -------------------- Model Classes --------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out, lengths):
        scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        mask = mask.to(lstm_out.device)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_matrix, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embed_text = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, lengths):
        x = self.embed_text(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        text_feat = self.attention(out, lengths)
        return self.fc(self.dropout(text_feat))

# -------------------- Load Embeddings and Model --------------------
embed_dim = 100
embed_matrix = torch.tensor(np.random.normal(scale=0.6, size=(len(vocab), embed_dim)), dtype=torch.float32)

model = LSTMWithAttention(
    vocab_size=len(vocab),
    embed_matrix=embed_matrix
)

MODEL_PATH = "fake_news_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1Guwx0RPy_7smL4iHXBST78j_AnIES4k_"  # Update if needed

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# -------------------- Prediction Function --------------------
def predict_fake_news(statement):
    tokens = torch.tensor([encode(tokenize(statement))])
    length = torch.tensor([len(tokens[0])])

    with torch.no_grad():
        output = model(tokens, length)
        prob = torch.softmax(output, dim=1)
        prediction = torch.argmax(prob, dim=1).item()
    return prediction, prob.numpy()
