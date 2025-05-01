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

# Dummy sample text list to create vocab (replace with actual vocab if available)
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

class LSTMWithMetadataAttention(nn.Module):
    def __init__(self, vocab_size, job_size, party_size, context_size, embed_matrix, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embed_text = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)
        self.embed_job = nn.Embedding(job_size, 16)
        self.embed_party = nn.Embedding(party_size, 8)
        self.embed_context = nn.Embedding(context_size, 8)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2 + 16 + 8 + 8, 2)

    def forward(self, x, lengths, job_id, party_id, context_id):
        x = self.embed_text(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        text_feat = self.attention(out, lengths)
        job_embed = self.embed_job(job_id)
        party_embed = self.embed_party(party_id)
        context_embed = self.embed_context(context_id)
        combined = torch.cat([text_feat, job_embed, party_embed, context_embed], dim=1)
        return self.fc(self.dropout(combined))

# -------------------- Load Embeddings and Model --------------------
embed_dim = 100
embed_matrix = torch.tensor(np.random.normal(scale=0.6, size=(len(vocab), embed_dim)), dtype=torch.float32)

model = LSTMWithMetadataAttention(
    vocab_size=len(vocab),
    job_size=14,
    party_size=6,
    context_size=14,
    embed_matrix=embed_matrix
)

MODEL_PATH = "fake_news_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1OqcWVx2BOnBIixfiE4PxUwbP2Dp4Xr_6"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# -------------------- Prediction Function --------------------
def predict_fake_news(statement, job, party, context):
    tokens = torch.tensor([encode(tokenize(statement))])
    length = torch.tensor([len(tokens[0])])
    job_tensor = torch.tensor([job])
    party_tensor = torch.tensor([party])
    context_tensor = torch.tensor([context])

    with torch.no_grad():
        output = model(tokens, length, job_tensor, party_tensor, context_tensor)
        prob = torch.softmax(output, dim=1)
        prediction = torch.argmax(prob, dim=1).item()
    return prediction, prob.numpy()
