print("Hello from Streamlit!")
import streamlit as st
import numpy as np
import math
import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from datasets import load_dataset

# ==========================================================
#                  BASIC UTILITIES
# ==========================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(text):
    return re.findall(r"\w+|[!?.]", str(text).lower())

def softmax(logits):
    a = np.array(logits)
    e = np.exp(a - np.max(a))
    return e / e.sum()

# ==========================================================
#                  LOAD DATASET
# ==========================================================
@st.cache_resource
def load_data():
    ds = load_dataset("mshenoda/spam-messages")
    cols = ds["train"].column_names

    text_col = "text" if "text" in cols else cols[0]
    label_map = {"ham": 0, "spam": 1, "0": 0, "1": 1}

    def normalize(example):
        lab = example.get("label")
        if isinstance(lab, (int, float)):
            example["label"] = int(lab)
        else:
            example["label"] = label_map.get(str(lab).lower(), 0)
        return example

    ds = ds.map(normalize)

    split = ds["train"].train_test_split(test_size=0.2, seed=SEED)
    train, test = split["train"], split["test"]

    return (
        [x[text_col] for x in train],
        [int(x["label"]) for x in train],
        [x[text_col] for x in test],
        [int(x["label"]) for x in test]
    )

# ==========================================================
#                 NAIVE BAYES IMPLEMENTATION
# ==========================================================
class NaiveBayes:
    def __init__(self):
        self.vocab = None
        self.log_spam = {}
        self.log_ham = {}
        self.P_spam = 0.5
        self.P_ham = 0.5
        self.unk_spam = 0.0
        self.unk_ham = 0.0

    def fit(self, texts, labels, alpha=1):
        vocab = set()
        for t in texts: vocab.update(tokenize(t))
        vocab = sorted(vocab)
        wc_spam, wc_ham = Counter(), Counter()

        spam_docs = sum(1 for l in labels if l == 1)
        ham_docs  = len(labels) - spam_docs
        total_docs = len(labels)

        for txt, lab in zip(texts, labels):
            toks = tokenize(txt)
            (wc_spam if lab == 1 else wc_ham).update(toks)

        self.P_spam = spam_docs / total_docs
        self.P_ham  = ham_docs  / total_docs

        V = len(vocab)
        total_spam = sum(wc_spam.values()) + alpha * V
        total_ham  = sum(wc_ham.values()) + alpha * V

        self.log_spam = {w: math.log((wc_spam[w] + alpha) / total_spam) for w in vocab}
        self.log_ham  = {w: math.log((wc_ham[w] + alpha) / total_ham)  for w in vocab}

        self.unk_spam = math.log(alpha / total_spam)
        self.unk_ham  = math.log(alpha / total_ham)
        self.vocab = set(vocab)

    def predict_proba(self, text):
        toks = tokenize(text)
        s_spam = math.log(self.P_spam + 1e-12)
        s_ham  = math.log(self.P_ham  + 1e-12)

        for t in toks:
            s_spam += self.log_spam.get(t, self.unk_spam)
            s_ham  += self.log_ham.get(t,  self.unk_ham)

        probs = softmax([s_ham, s_spam])
        return probs

# ==========================================================
#                   CNN MODEL
# ==========================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 100, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(100, 100, 3),
            nn.Conv1d(100, 100, 4),
            nn.Conv1d(100, 100, 5)
        ])
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        x = self.embed(x).transpose(1,2)
        pools = [torch.relu(c(x)).max(2)[0] for c in self.convs]
        return self.fc(torch.cat(pools, 1))

# ==========================================================
#                   BiLSTM MODEL
# ==========================================================
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(h)

# ==========================================================
#         VOCAB + ENCODING FUNCTION
# ==========================================================
@st.cache_resource
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t in texts:
        for tok in tokenize(t):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

MAX_LEN = 40

def encode(text, vocab):
    toks = tokenize(text)
    ids = [vocab.get(t, 1) for t in toks[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))

# ==========================================================
#                 TRAIN ALL MODELS
# ==========================================================
@st.cache_resource
def train_all_models():
    train_texts, train_labels, _, _ = load_data()

    # ---- NB ----
    nb = NaiveBayes()
    nb.fit(train_texts, train_labels)

    # ---- LR ----
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(train_texts)

    extra = np.array([[
        len(t), t.count("!"), sum(c.isdigit() for c in t),
        sum(c.isupper() for c in t),
        len(re.findall(r"http|www|\.com", t.lower())),
        sum(k in t.lower() for k in ["free", "win", "cash", "urgent", "click"])
    ] for t in train_texts])

    X_lr = hstack([X_train_tfidf, csr_matrix(extra)])
    lr = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)
    lr.fit(X_lr, train_labels)

    # ---- Vocab ----
    vocab = build_vocab(train_texts)

    # ---- CNN ----
    cnn = TextCNN(len(vocab)).to(device)
    cnn_opt = optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    encoded = [encode(t, vocab) for t in train_texts]
    ds = list(zip(encoded, train_labels))

    cnn.train()
    for epoch in range(1):
        np.random.shuffle(ds)
        for i in range(0, len(ds), 64):
            batch = ds[i:i+64]
            Xb = torch.tensor([b[0] for b in batch], dtype=torch.long).to(device)
            yb = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)

            cnn_opt.zero_grad()
            loss = loss_fn(cnn(Xb), yb)
            loss.backward()
            cnn_opt.step()
    cnn.eval()

    # ---- BiLSTM ----
    lstm = BiLSTM(len(vocab)).to(device)
    lstm_opt = optim.Adam(lstm.parameters(), lr=1e-3)

    lstm.train()
    for epoch in range(1):
        np.random.shuffle(ds)
        for i in range(0, len(ds), 64):
            batch = ds[i:i+64]
            Xb = torch.tensor([b[0] for b in batch], dtype=torch.long).to(device)
            yb = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            lstm_opt.zero_grad()
            loss = loss_fn(lstm(Xb), yb)
            loss.backward()
            lstm_opt.step()
    lstm.eval()

    return {
        "nb": nb,
        "lr": lr,
        "tfidf": tfidf,
        "cnn": cnn,
        "lstm": lstm,
        "vocab": vocab
    }

models = train_all_models()

# ==========================================================
#           PREDICT ACROSS ALL FOUR MODELS
# ==========================================================
def predict_all_models(text):
    results = []
    vocab = models["vocab"]

    # NB
    probs = models["nb"].predict_proba(text)
    results.append(("Naive Bayes", int(np.argmax(probs)), float(max(probs)), probs.tolist()))

    # LR
    tfidf = models["tfidf"]
    lr = models["lr"]
    feat = tfidf.transform([text])
    extra = np.array([[len(text), text.count("!"), sum(c.isdigit() for c in text), sum(c.isupper() for c in text), len(re.findall(r"http|www|\.com", text.lower())), sum(k in text.lower() for k in ["free", "win", "cash", "urgent", "click"])]])
    Xlr = hstack([feat, csr_matrix(extra)])
    probs = lr.predict_proba(Xlr)[0]
    results.append(("Logistic Regression", int(np.argmax(probs)), float(max(probs)), probs.tolist()))

    # CNN
    ids = torch.tensor([encode(text, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = models["cnn"](ids).cpu().numpy()[0]
        probs = softmax(logits)
    results.append(("CNN", int(np.argmax(probs)), float(max(probs)), probs.tolist()))

    # LSTM
    with torch.no_grad():
        logits = models["lstm"](ids).cpu().numpy()[0]
        probs = softmax(logits)
    results.append(("BiLSTM", int(np.argmax(probs)), float(max(probs)), probs.tolist()))

    return results

# ==========================================================
#                  STREAMLIT UI
# ==========================================================

st.set_page_config(page_title="AI Spam Detector", page_icon="📩", layout="centered")

# Styling
st.markdown("""
<style>
.main-title { font-size: 42px; font-weight: 900; text-align: center; color: #4A90E2; }
.sub-text { text-align: center; font-size: 18px; color: #555; }
.result-box { padding: 20px; border-radius: 12px; background: #f7f9fc; border: 1px solid #d8e3f0; margin-bottom: 20px; }
.winner { padding: 25px; border-radius: 15px; background: #E3F7E0; border: 2px solid #9ED89E; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📩 AI Spam Detection Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Test one message across all four models and pick the most confident one.</div>', unsafe_allow_html=True)

st.markdown("---")

text = st.text_area("✍️ Enter a message to classify:", height=160)

if st.button("🚀 Classify Message", use_container_width=True):
    if not text.strip():
        st.warning("Please type a message first.")
    else:
        with st.spinner("Running all models..."):
            results = predict_all_models(text)

        st.subheader("📊 Model Results")

        for name, pred, conf, probs in results:
            st.markdown(f"<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"### 🔹 {name}")
            st.write(f"Prediction: **{pred}**  (0=ham, 1=spam)")
            st.write(f"Confidence: **{conf:.4f}**")
            st.write(f"Probabilities → Ham: {probs[0]:.4f} | Spam: {probs[1]:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        best = max(results, key=lambda x: x[2])

        st.markdown("<div class='winner'>", unsafe_allow_html=True)
        st.markdown(f"## 🏆 Best Model: **{best[0]}**")
        st.write(f"Final Prediction: **{best[1]}**")
        st.write(f"Highest Confidence: **{best[2]:.4f}**")
        st.markdown("</div>", unsafe_allow_html=True)
