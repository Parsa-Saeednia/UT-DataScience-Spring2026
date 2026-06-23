import pandas as pd

def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['duration'] >= 60.0]
    if 'canonical_composer' in df.columns:
        composer_counts = df['canonical_composer'].value_counts()
        viable = composer_counts[composer_counts >= 10].index
        df = df[df['canonical_composer'].isin(viable)]
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'duration' in df.columns and not df.empty:
        train_slice = df[df['split'] == 'train'] if 'split' in df.columns else df
        if train_slice.empty:
            train_slice = df
            
        mean_duration = train_slice['duration'].mean()
        std_duration = train_slice['duration'].std()
        
        df['duration_scaled'] = (df['duration'] - mean_duration) / std_duration if std_duration > 0 else 0.0
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_noise(df)
    df = scale_features(df)
    return df