from model import Tokenizer, Transformer
from data import FakePhoneDataset
from torch.utils.data import DataLoader
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
import wandb

dataset = pickle.load(open("phonebook.pkl", "rb"))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

val = FakePhoneDataset(num_people=50, num_samples=32)
val_dataloader = DataLoader(val, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = Tokenizer("./tokenizer.model")

model = Transformer(
    embed_size=256,
    vocab_size=32000,
    num_layers=4,
    heads=8,
    dropout=0.1,
    forward_expansion=512,
    device=device
)

model.to(device)

print(sum(p.numel() for p in model.parameters()))

exit()

optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10

wandb.init(project="phonebook_hallucinations", entity="pranavnt", config={
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 10
})

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        batch = tokenizer.batch_encode(batch).to(device)
        print(batch.shape)
        inputs, labels = batch[:, :-1], batch[:, 1:]
        print(inputs.shape)
        print(labels.shape)

        out = model(inputs)
        print(out.shape)
        out = out.view(-1, out.size(-1))
        labels = labels.contiguous().view(-1)
        loss = F.cross_entropy(out, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 50 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss.item()
            })

        if batch_idx % 250 == 0:
            torch.save(model.state_dict(), f"./checkpoint/phonebook_transformer_{epoch}_{batch_idx}.pt")

    for batch_val in val_dataloader:
        batch_val = tokenizer.batch_encode(batch_val).to(device)
        out = model(batch_val[:, :-1])
        out = out.view(-1, out.size(-1))
        labels_val = batch_val[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(out, labels_val)

        wandb.log({
            "epoch": epoch,
            "val_loss": loss.item()
        })

    torch.save(model.state_dict(), f"./checkpoint/phonebook_transformer_{epoch}.pt")
    wandb.save(f"./checkpoint/phonebook_transformer_{epoch}.pt")