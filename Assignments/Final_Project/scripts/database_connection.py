import os
from typing import Dict, Any
from sqlalchemy import create_engine
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
