"""create documents and chunks tables

Revision ID: 20250613_create_documents_and_chunks
Revises: 
Create Date: 2025-06-13 06:17:00.000000

"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision = '20250613_create_documents_and_chunks'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('title', sa.String, index=True),
        sa.Column('file_path', sa.String, nullable=True),
        sa.Column('doc_metadata', sa.JSON, nullable=True),
        pgvector.sqlalchemy.Vector(384, name='embedding', nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    op.create_table(
        'chunks',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('document_id', sa.Integer, sa.ForeignKey('documents.id'), index=True),
        sa.Column('text', sa.String),
        sa.Column('page_number', sa.Integer, nullable=True),
        sa.Column('chunk_type', sa.String, nullable=True),
        pgvector.sqlalchemy.Vector(384, name='embedding', nullable=True),
        sa.Column('doc_metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False),
    )
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

def downgrade():
    op.drop_table('chunks')
    op.drop_table('documents')