"""add minio_key and neo4j_node_id to documents

Revision ID: 20250616_add_minio_and_neo4j_to_documents
Revises: '20250613061700'
Create Date: 2025-06-16 16:45:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20250616164500'
down_revision = '20250613061700'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('documents', sa.Column('minio_key', sa.String(length=255), nullable=True))
    op.add_column('documents', sa.Column('neo4j_node_id', sa.String(length=128), nullable=True))

def downgrade():
    op.drop_column('documents', 'neo4j_node_id')
    op.drop_column('documents', 'minio_key')