import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class MusicDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, seq_length: int = 50):
        self.seq_length = seq_length
        print(f"[Dataset] Building sequences with seq_length={seq_length} ...")
        self.sequences = self._build_sequences(dataframe)
        print(f"[Dataset] Finished. Total sequences: {len(self.sequences)}")

    def _build_sequences(self, df: pd.DataFrame) -> list:
        seqs = []
        for idx, tokens in enumerate(df['sequence_tokens']):
            if isinstance(tokens, list) and len(tokens) > self.seq_length:
                for i in range(len(tokens) - self.seq_length):
                    seqs.append((tokens[i:i+self.seq_length], tokens[i+self.seq_length]))
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y[0] - 60, dtype=torch.long)

class MusicLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def evaluate_model(model, dataloader, criterion, device):
    print("[Eval] Starting validation...")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            if batch_idx == 0:
                print(f"[Eval] First validation batch - X shape: {batch_x.shape}, Y shape: {batch_y.shape}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[Eval] Validation finished. Avg loss: {avg_loss:.4f}")
    return avg_loss

def execute_training() -> None:
    print("[Main] Training started...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataframes', 'final_features.pkl')
    model_path = os.path.join(base_dir, 'models', 'music_lstm.pth')

    print(f"[Main] Looking for dataset at: {data_path}")
    print(f"[Main] Model will be saved to: {model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")

    print("[Main] Loading dataframe...")
    df = pd.read_pickle(data_path)
    print(f"[Main] Dataframe loaded. Rows: {len(df)}")

    dataset = MusicDataset(df, seq_length=50)
    print(f"[Main] Dataset size: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"[Main] Splitting dataset -> Train: {train_size}, Validation: {val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    input_dim = 17
    vocab_size = 36
    hidden_dim = 128
    num_layers = 3
    lr = 0.0005
    final_epochs = 10

    print("[Main] Creating model...")
    model = MusicLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("[Main] Starting training loop...")
    for epoch in range(final_epochs):
        print(f"\n[Train] Epoch {epoch+1}/{final_epochs} started")
        model.train()
        total_train_loss = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if epoch == 0 and batch_idx == 0:
                print(f"[Train] First training batch - X shape: {batch_x.shape}, Y shape: {batch_y.shape}")

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)
        avg_train_loss = total_train_loss / len(train_loader)

        print(f"[Train] Epoch {epoch+1} finished | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("[Main] Saving model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[Main] Model saved successfully at: {model_path}")

if __name__ == "__main__":
    execute_training()