import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import os
torch.cuda.empty_cache()

# Define a function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.TransformerEncoderLayer):
        nn.init.xavier_uniform_(m.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(m.linear1.weight)
        nn.init.xavier_uniform_(m.linear2.weight)

# Load configuration from the JSON file
data_Set = ['BTC', 'SPX', 'TAIEX']
for d in data_Set:
    config_path = "C:\\gitproject\\tradebot\\ML\\btcDif\\config"+d+"more.json"
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Load data from the CSV file
    data_path = config["file_path"]
    df = pd.read_csv(data_path)

    # Count the number of null values in each column
    # print(df.isnull().sum())
    # Drop rows with missing values
    features = config["features"].replace("'", "").replace(", ", ",").split(",")
    data = df[features].values
    # Use Min-Max Scaling to scale the data to the [0, 1] range
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    data = (data - min_value) / (max_value - min_value + 1e-8)  # Added epsilon to avoid division by zero

    # Split the data into train and test sets
    train_percent = config["train_Percent"]
    train_size = int(len(data) * train_percent)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Dataset class to handle the input sequence data
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, seq_length):
            self.data = data
            self.seq_length = seq_length

        def __len__(self):
            return len(self.data) - self.seq_length - 1

        def __getitem__(self, index):
            x = self.data[index:index+self.seq_length]
            # Predicting next change rate (close / close[-1]) with safe divide
            prev_value = self.data[index + self.seq_length, 0]
            next_value = self.data[index + self.seq_length + 1, 0]
            y = next_value / (prev_value + 1e-8)  # Added epsilon to avoid division by zero
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # Prepare datasets and dataloaders for training and testing
    seq_length = config["seq_len"]

    train_dataset = TimeSeriesDataset(train_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Transformer model definition
    class TransformerPredictor(nn.Module):
        def __init__(self, input_dim, seq_length, num_heads, num_layers, hidden_dim):
            super(TransformerPredictor, self).__init__()
            self.input_dim = input_dim
            self.seq_length = seq_length
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc_price = nn.Linear(hidden_dim, 1)  # Predict price change
            self.fc_confidence = nn.Linear(hidden_dim, 1)  # Predict confidence
            self.sigmoid = nn.Softmax(dim=1)  # Apply softmax to get confidence level between 0 and 1
            # self.sigmoid = nn.Sigmoid()  # Apply sigmoid to get confidence level between 0 and 1
        def forward(self, x):
            # Create embeddings with positional encodings
            x = self.embedding(x) + self.pos_encoder
            x = self.transformer(x)  # No need to permute if batch_first=True
            x = x[:, -1, :]  # Get the last time step
            price_change = self.fc_price(x)
            confidence = self.sigmoid(self.fc_confidence(x))  # Apply sigmoid to get confidence level between 0 and 1
            return price_change, confidence

    # Parameters from config file
    input_dim = len(features)
    hidden_dim = config["hidden_dim"]
    num_heads = config["nhead"]
    num_layers = config["num_layers"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    # Model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerPredictor(input_dim, seq_length, num_heads, num_layers, hidden_dim).to(device)

    # Initialize weights
    model.apply(init_weights)

    criterion = nn.SmoothL1Loss()  # Changed to Smooth L1 Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler()

    model_save_path = config["save_path"] + config["model_name"] + ".pth"
    best_model_save_path = model_save_path.replace('.pth', '_best.pth')
    best_loss = float('inf')  # Initial best loss set to infinity

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', best_loss)
        print(f'Resuming training from epoch {start_epoch}')

    # Generalized Kelly Criterion function
    def generalized_kelly(predicted_price, confidence, leverage=10000.0):
        expected_return = torch.sum(predicted_price * confidence)
        variance = torch.sum((predicted_price - expected_return) ** 2 * confidence)
       
        kelly_fraction = expected_return / variance
        return kelly_fraction*leverage

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda'):
                predicted_price, confidence = model(x)
                loss = criterion(predicted_price.squeeze(), y)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.16f}')
        scheduler.step(avg_loss)

        # Save current model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, model_save_path)

        # Save the best model if the current loss is the best
        model.eval()
        test_total_loss = 0
        with torch.no_grad():
            balance = 1000.0  # Initial capital in USD
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                with torch.amp.autocast(device_type='cuda'):
                    predicted_price, confidence = model(x)
                    loss = criterion(predicted_price.squeeze(), y)
                    prev_price = x[0, -1, 0].item()  # Previous price
                    current_price = y[0].item()  # Current price
                    delta = current_price - prev_price
                    kelly_fraction = generalized_kelly(predicted_price.squeeze(), confidence.squeeze())
                    balance += balance / prev_price * kelly_fraction * delta
                    
                test_total_loss += loss.item()

        avg_test_loss = test_total_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.16f}, Balance: {balance:.4f}")
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved with loss          {best_loss:.16f}")
        model.train()
    # Testing loop
    model.eval()
    test_total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type='cuda'):
                predicted_price, confidence = model(x)
                loss = criterion(predicted_price.squeeze(), y)
                
                # Calculating probability distribution and optimal fraction using Kelly criterion
                kelly_fraction = generalized_kelly(predicted_price.squeeze(), confidence.squeeze())
                print(f"Optimal Kelly Fraction: {kelly_fraction.item():.4f}")

            test_total_loss += loss.item()

    avg_test_loss = test_total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.16f}")

    print("Training complete.")


