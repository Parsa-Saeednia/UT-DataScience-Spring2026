import os
import pandas as pd
from database_connection import get_db_engine

def load_csv_data() -> pd.DataFrame:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    csv_file_path = os.path.join(project_root, 'data', 'maestro-v3.0.0', 'maestro-v3.0.0.csv')
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Dataset not found: {csv_file_path}")
        
    return pd.read_csv(csv_file_path)

def main():
    df = load_csv_data()
    engine = get_db_engine()
    
    composers_df = df[['canonical_composer']].drop_duplicates().rename(columns={'canonical_composer': 'name'})
    composers_df.to_sql('composers', con=engine, if_exists='append', index=False)
    
    composers_db = pd.read_sql("SELECT id as composer_id, name as canonical_composer FROM composers", con=engine)
    df = df.merge(composers_db, on='canonical_composer')
    
    pieces_df = df[['composer_id', 'canonical_title']].drop_duplicates().rename(columns={'canonical_title': 'title'})
    pieces_df.to_sql('pieces', con=engine, if_exists='append', index=False)
    
    pieces_db = pd.read_sql("SELECT id as piece_id, composer_id, title as canonical_title FROM pieces", con=engine)
    df = df.merge(pieces_db, on=['composer_id', 'canonical_title'])
    
    performances_df = df[['piece_id', 'split', 'year', 'duration', 'midi_filename', 'audio_filename']]
    performances_df.to_sql('performances', con=engine, if_exists='append', index=False)

if __name__ == "__main__":
    main()