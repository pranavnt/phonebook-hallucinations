import sentencepiece as spm
import os

# Set the path to your training data file
train_data_path = "phonebook.txt"

# Set the desired vocabulary size
vocab_size = 1024

# Set the output path for the tokenizer model
model_prefix = "tokenizer"

# Train the tokenizer
spm.SentencePieceTrainer.Train(
    input=train_data_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# Print the path to the generated tokenizer files
print("Tokenizer model file:", model_prefix + ".model")
print("Tokenizer vocab file:", model_prefix + ".vocab")