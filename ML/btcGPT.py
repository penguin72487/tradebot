import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import pandas as pd

# Load and preprocess K-line data
class KlineDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_data = self.prepare_text_data()

    def prepare_text_data(self):
        descriptions = []
        for _, row in self.data.iterrows():
            description = f"{row['time']} {row['open']} {row['high']} {row['low']} {row['close']} {row['Volume']}"
            descriptions.append(description)
        return descriptions

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        encoded_data = self.tokenizer(
            self.text_data[idx],
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        return encoded_data.input_ids.squeeze(), encoded_data.attention_mask.squeeze()

# Set up configurations
file_path = 'C:\\gitproject\\tradebot\\ML\\BITSTAMP_BTCUSD, 240.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "gpt2"
learning_rate = 5e-5
epochs = 100
batch_size = 8
max_length = 512

# Load tokenizer and dataset
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add pad token if not present
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dataset = KlineDataset(file_path, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create GPT2 model from scratch
config = GPT2Config(n_layer=12, n_head=12, n_embd=768)
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))  # Adjust token embeddings for new pad token
model = model.to(device)

# Set up optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}")

# Save the trained model
model.save_pretrained('./kline_gpt_model')
tokenizer.save_pretrained('./kline_gpt_model')

print("Model training completed and saved.")
