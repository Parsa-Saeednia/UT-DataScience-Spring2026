import os
from typing import Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

def get_db_config() -> Dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "name": os.getenv("DB_NAME")
    }

def get_server_engine() -> Engine:
    config = get_db_config()
    server_url = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/"
    return create_engine(server_url)

def get_db_engine() -> Engine:
    config = get_db_config()
    db_url = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    return create_engine(db_url)

def setup_database() -> None:
    config = get_db_config()
    engine = get_server_engine()
    
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {config['name']}"))

def execute_schema() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    schema_path = os.path.join(project_root, 'database', 'schema.sql')
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as file:
        sql_commands = file.read().split(';')
        
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        for command in sql_commands:
            clean_command = command.strip()
            if clean_command:
                conn.execute(text(clean_command))

def main():
    setup_database()
    execute_schema()

if __name__ == "__main__":
    main()