import os
import pandas as pd
from sqlalchemy import text
from database_connection import get_db_engine

def read_query(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Query file missing at: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def execute_query(query: str) -> pd.DataFrame:
    with get_db_engine().connect() as conn:
        return pd.read_sql(text(query), con=conn)

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_pickle(path)

def load_data_from_db() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    query_path = os.path.join(base_dir, 'database', 'load_query.sql')
    pkl_path = os.path.join(base_dir, 'data', 'dataframes', 'raw_data.pkl')
    
    sql_query = read_query(query_path)
    df = execute_query(sql_query)
    save_dataframe(df, pkl_path)

if __name__ == "__main__":
    load_data_from_db()