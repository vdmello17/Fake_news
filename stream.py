import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Set max sequence length (must match training)
MAX_LEN = 100

# 1) Define the LSTMWithAttention model class
class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float), freeze=False
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        attn_scores = self.attn(output).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        return self.fc(context), attn_weights

# 2) Load tokenizer, embedding matrix, and model
@st.cache_resource
def load_components():
    # Load tokenizer JSON
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tok_json = f.read()
    tokenizer = tokenizer_from_json(tok_json)

    # Load embedding matrix and determine dims
    embed_matrix = np.load("glove_embedding_matrix.npy")
    vocab_size, embed_dim = embed_matrix.shape

    # Instantiate and load model (must match training args)
    hidden_dim = 128
    output_dim = 2
    model = LSTMWithAttention(vocab_size, embed_dim, hidden_dim, output_dim, embed_matrix)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return tokenizer, model

# Initialize components
tokenizer, model = load_components()

# 3) Streamlit UI
st.title("📰 Fake News Detector")
# Add threshold slider for tuning sensitivity on Fake class
st.write("Adjust the threshold for classifying 'Fake'. If the Fake probability is above this value, the prediction will be 'Fake'.")
threshold = st.slider(
    "Fake probability threshold", min_value=0.0, max_value=1.0,
    value=0.5, step=0.01
)
