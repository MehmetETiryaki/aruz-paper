import importlib.util
import json
import os
import random
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eval_meter", ROOT / "scripts/eval_meter.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)

class Net(nn.Module):
    def __init__(self, vocab, classes, emb=64, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, classes)

    def forward(self, x, lengths):
        h, _ = self.rnn(nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True,
                                                          enforce_sorted=False))
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (x[:, :h.size(1)] != 0).unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1)
        mx = h.masked_fill(~mask, -1e9).max(1).values
        return self.out(self.drop(torch.cat([mean, mx], -1)))

def lines_of(recs):
    out = []
    for r in recs:
        for u in r["units"]:
            for m in u["misras"]:
                out.append((r["id"], " ".join(em.words(m, r)).lower(), r["meter"]))
    return out

def batches(data, chars, classes, size, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for k in range(0, len(idx), size):
        part = [data[i] for i in idx[k:k + size]]
        seqs = [[chars.get(c, 1) for c in t] or [1] for _, t, _ in part]
        n = max(map(len, seqs))
        x = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        y = torch.tensor([classes[m] for _, _, m in part])
        yield x, torch.tensor([len(s) for s in seqs]), y, part

def predict(net, data, chars, classes):
    net.eval()
    out = []
    with torch.no_grad():
        for x, lengths, _, part in batches(data, chars, classes, 256, False, None):
            out += torch.log_softmax(net(x, lengths), -1).tolist()
    return out
