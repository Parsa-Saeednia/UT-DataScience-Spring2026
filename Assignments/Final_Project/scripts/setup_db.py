import os
import pandas as pd
from sqlalchemy import text
import unicodedata
from database_connection import get_db_config, get_server_engine, get_db_engine

def setup_database() -> None:
    print("Creating database if it doesn't exist...")
    config = get_db_config()
    with get_server_engine().connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {config['name']}"))

def execute_schema() -> None:
    print("Executing schema.sql...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, 'database', 'schema.sql')
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as file:
        sql_commands = file.read().split(';')
        
    with get_db_engine().connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        for command in [c.strip() for c in sql_commands if c.strip()]:
            conn.execute(text(command))

def load_and_clean_data() -> pd.DataFrame:
    print("Loading and deeply normalizing CSV data...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'maestro-v3.0.0', 'maestro-v3.0.0.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    for col in ['canonical_composer', 'canonical_title']:
        df[col] = df[col].astype(str).apply(lambda x: unicodedata.normalize('NFKD', x))
        
        df[col] = df[col].str.encode('ascii', errors='ignore').str.decode('utf-8')
        
        df[col] = df[col].str.replace("’", "'").str.replace("`", "'")
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip().str.title()
    
    return df

def insert_composers(df: pd.DataFrame, engine) -> pd.DataFrame:
    print("Inserting composers...")
    composers_df = df[['canonical_composer']].drop_duplicates().rename(columns={'canonical_composer': 'name'})
    composers_df.to_sql('composers', con=engine, if_exists='append', index=False)
    
    composers_db = pd.read_sql("SELECT id as composer_id, name as canonical_composer FROM composers", con=engine)
    return df.merge(composers_db, on='canonical_composer')

def insert_pieces(df: pd.DataFrame, engine) -> pd.DataFrame:
    print("Inserting pieces...")
    pieces_df = df[['composer_id', 'canonical_title']].drop_duplicates().rename(columns={'canonical_title': 'title'})
    pieces_df.to_sql('pieces', con=engine, if_exists='append', index=False)
    
    pieces_db = pd.read_sql("SELECT id as piece_id, composer_id, title as canonical_title FROM pieces", con=engine)
    return df.merge(pieces_db, on=['composer_id', 'canonical_title'])

def insert_performances(df: pd.DataFrame, engine) -> None:
    print("Inserting performances...")
    performances_df = df[['piece_id', 'split', 'year', 'duration', 'midi_filename', 'audio_filename']]
    performances_df.to_sql('performances', con=engine, if_exists='append', index=False)

def import_data() -> None:
    df = load_and_clean_data()
    engine = get_db_engine()
    
    df = insert_composers(df, engine)
    df = insert_pieces(df, engine)
    insert_performances(df, engine)

def main():
    print("Starting Database Setup...")
    setup_database()
    execute_schema()
    import_data()
    print("Database Setup Complete!")

if __name__ == "__main__":
    main()