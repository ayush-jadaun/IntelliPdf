from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# If your models live in backend/app/core/database/models.py, adjust this path as needed:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'core', 'database')))
from models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set this to your SQLAlchemy Base metadata for 'alembic revision --autogenerate' to work!
target_metadata = Base.metadata

def get_url():
    return "postgresql+psycopg2://intellipdf:securepass@postgres:5432/intellipdf_db"

def run_migrations_offline():
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()