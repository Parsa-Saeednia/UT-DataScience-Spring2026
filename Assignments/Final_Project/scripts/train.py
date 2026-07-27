import os
import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================
# 1. Dataset Definition
# ==========================================
class MusicDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, seq_length: int = 50):
        self.seq_length = seq_length
        self.sequences = self._build_sequences(dataframe)

    def _build_sequences(self, df: pd.DataFrame) -> list:
        seqs = []
        for tokens in df['sequence_tokens']:
            if isinstance(tokens, list) and len(tokens) > self.seq_length:
                for i in range(len(tokens) - self.seq_length):
                    # Input sequence and the target token
                    seqs.append((tokens[i:i+self.seq_length], tokens[i+self.seq_length]))
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        # x has 17 features per timestep. y[0] is the target pitch to predict.
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y[0], dtype=torch.long)

# ==========================================
# 2. Model Architecture
# ==========================================
class MusicLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Extract the output of the last timestep
        return out

# ==========================================
# 3. Training & Evaluation Routines
# ==========================================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_configuration(params: dict, train_loader, val_loader, input_dim: int, vocab_size: int, device: str) -> float:
    """Trains a model with specific hyperparameters for a few epochs to evaluate its potential."""
    model = MusicLSTM(
        input_dim=input_dim, 
        hidden_dim=params['hidden_dim'], 
        num_layers=params['num_layers'], 
        vocab_size=vocab_size
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    epochs_for_search = 2  # Keep this low for faster hyperparameter search
    for epoch in range(epochs_for_search):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    val_loss = evaluate_model(model, val_loader, criterion, device)
    return val_loss

# ==========================================
# 4. Greedy Hyperparameter Optimization
# ==========================================
def greedy_hyperparameter_search(train_loader, val_loader, input_dim: int, vocab_size: int, device: str) -> dict:
    print("\n--- Starting Greedy Hyperparameter Search ---")
    
    # Define the search space
    param_space = {
        'hidden_dim': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'lr': [0.005, 0.001, 0.0005]
    }
    
    # Set a baseline starting point
    best_params = {'hidden_dim': 128, 'num_layers': 2, 'lr': 0.001}
    print(f"Baseline Params: {best_params}")
    best_loss = train_configuration(best_params, train_loader, val_loader, input_dim, vocab_size, device)
    print(f"Baseline Validation Loss: {best_loss:.4f}\n")
    
    # Iterate through parameters greedily
    for param_name, search_values in param_space.items():
        for value in search_values:
            if value == best_params[param_name]:
                continue
                
            test_params = best_params.copy()
            test_params[param_name] = value
            
            print(f"Testing {param_name} = {value} (Holding others at {best_params})")
            val_loss = train_configuration(test_params, train_loader, val_loader, input_dim, vocab_size, device)
            print(f"Resulting Loss: {val_loss:.4f}")
            
            # Greedy update: If it improves, keep it immediately for subsequent tests
            if val_loss < best_loss:
                print(f"-> Improvement found! Updating best {param_name} to {value}\n")
                best_loss = val_loss
                best_params = test_params
            else:
                print("-> No improvement.\n")
                
    print(f"--- Search Complete. Optimal Parameters: {best_params} ---")
    return best_params

# ==========================================
# 5. Main Execution
# ==========================================
def execute_training() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataframes', 'final_features.pkl')
    model_path = os.path.join(base_dir, 'models', 'music_lstm.pth')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")
        
    print("Loading prepared feature dataset...")
    df = pd.read_pickle(data_path)
    
    dataset = MusicDataset(df, seq_length=50)
    
    # Create Train/Validation Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Adjust num_workers based on your CPU capability
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = 17  # 5 standard features + 12 harmonic context features
    vocab_size = 128  # 0-127 MIDI pitches
    
    # 1. Run Greedy Search to find best hyperparameters
    best_params = greedy_hyperparameter_search(train_loader, val_loader, input_dim, vocab_size, device)
    
    # 2. Train Final Model using the best parameters
    print("\nTraining Final Model with optimal parameters...")
    final_model = MusicLSTM(
        input_dim=input_dim, 
        hidden_dim=best_params['hidden_dim'], 
        num_layers=best_params['num_layers'], 
        vocab_size=vocab_size
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
    
    final_epochs = 10
    for epoch in range(final_epochs):
        final_model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        val_loss = evaluate_model(final_model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{final_epochs} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(final_model.state_dict(), model_path)
    print(f"\nFinal model successfully saved to {model_path}")

if __name__ == "__main__":
    execute_training()