import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd

# Load trained model and tokenizer
model_path = './kline_gpt_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Load historical data for simulation
file_path = 'C:\\gitproject\\tradebot\\ML\\BITSTAMP_BTCUSD, 240.csv'
data = pd.read_csv(file_path)

# Initialize portfolio
cash = 10000  # Initial cash in USD
position = 0  # Number of units of BTC held
transaction_log = []

# Simulation loop
for idx, row in data.iterrows():
    # Print progress of the simulation
    if idx % 10 == 0:
        print(f"Simulating step {idx}/{len(data)}")

    # Prepare input
    description = f"{row['time']} {row['open']} {row['high']} {row['low']} {row['close']} {row['Volume']}"
    encoded_input = tokenizer(
        description, 
        return_tensors='pt', 
        max_length=512, 
        truncation=True, 
        padding='max_length'
    )

    # Get model prediction with attention mask and adjusted settings
    with torch.no_grad():
        output = model.generate(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'],  # Add attention mask
            max_new_tokens=50,  # Control the number of tokens generated
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
        )
    
    # Decode output to understand trading action
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    
    
    # Determine trading action based on prediction (simplified logic)
    if '買入' in prediction and cash > row['close']:
        # Buy as much BTC as possible
        amount_to_buy = cash // row['close']
        position += amount_to_buy
        cash -= amount_to_buy * row['close']
        transaction_log.append(f"Buy {amount_to_buy} BTC at {row['close']} on {row['time']}")
    elif '賣出' in prediction and position > 0:
        # Sell all BTC held
        cash += position * row['close']
        transaction_log.append(f"Sell {position} BTC at {row['close']} on {row['time']}")
        position = 0


# Calculate final portfolio value
final_value = cash + position * data.iloc[-1]['close']
print(f"Final portfolio value: ${final_value:.2f}")

# Output transaction log
for log in transaction_log:
    print(log)
