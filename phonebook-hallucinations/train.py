from model import Tokenizer, Transformer
from data import FakePhoneDataset
from torch.utils.data import DataLoader
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb

dataset = pickle.load(open("phonebook.pkl", "rb"))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = Tokenizer("./tokenizer.model")
model = Transformer(
    embed_size=512,
    vocab_size=32000,
    num_layers=6,
    heads=8,
    dropout=0.1,
    forward_expansion=1024,
    device=device
)

model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10

wandb.init(project="hallucinations_memory_safety", entity="pranavnt")

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        batch = tokenizer.batch_encode(batch).to(device)

        out = model(batch)

        loss = F.cross_entropy(out, batch)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Batch: ", batch_idx)
