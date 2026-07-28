import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
# Import the architecture and dataset classes from your train script
from train import MusicLSTM, MusicDataset 

def evaluate_model():
    print("\n--- Starting Model Evaluation ---")
    
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataframes', 'final_features.pkl')
    model_path = os.path.join(base_dir, 'models', 'music_lstm.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the evaluation data
    print("Loading test data...")
    df = pd.read_pickle(data_path)
    
    # In a production environment, you should use a strictly hold-out test set here.
    # For demonstration, we are evaluating on the available dataset structure.
    dataset = MusicDataset(df, seq_length=50) 
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 3. Initialize the architecture and load the weights
    print("Loading model weights...")
    input_dim = 17 
    vocab_size = 128
    
    # Note: These hyperparameters MUST match the best_params found during training
    # For a fully automated pipeline, you would save best_params to a JSON and load them here.
    # We will assume the defaults or a known good configuration for now.
    model = MusicLSTM(input_dim=input_dim, hidden_dim=256, num_layers=3, vocab_size=vocab_size)
    
    # Load the state dictionary (the learned weights)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # 4. Run Evaluation
    print("Evaluating predictions...")
    correct_predictions = 0
    total_predictions = 0
    
    # Disable gradient calculation for evaluation to save memory and compute
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            
            # The output is a probability distribution over the vocab size. 
            # We take the argmax to find the predicted pitch.
            _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()
            
    accuracy = 100 * correct_predictions / total_predictions
    print(f"====================================")
    print(f"Model Pitch Prediction Accuracy: {accuracy:.2f}%")
    print(f"====================================")

if __name__ == "__main__":
    evaluate_model()