import os
import pandas as pd
from database_connection import get_db_engine

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file missing at: {path}")
    return pd.read_pickle(path)

def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ['year', 'audio_filename', 'canonical_title'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    if 'canonical_composer' in df.columns and not df.empty:
        df['composer_encoded'] = df['canonical_composer'].astype('category').cat.codes
    return df

def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)

def preprocess_music_pipeline() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_pkl = os.path.join(base_dir, 'data', 'dataframes', 'raw_data.pkl')
    preprocessed_pkl = os.path.join(base_dir, 'data', 'dataframes', 'preprocessed_data.pkl')
    
    df = load_data(raw_pkl)
    df = clean_and_encode(df)
    save_data(df, preprocessed_pkl)

if __name__ == "__main__":
    preprocess_music_pipeline()