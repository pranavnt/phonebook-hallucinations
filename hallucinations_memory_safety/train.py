from model import Tokenizer, Transformer
from torch.utils.data import DataLoader
import pickle
import wandb

dataset = pickle.load(open("phonebook.pkl", "rb"))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

tokenizer = Tokenizer("./tokenizer.model")

for batch_idx, batch in enumerate(dataloader):

