"""groq-only embeddings: wipe stored vectors, switch to 768 dims, add HNSW

The previous 3072-dim Gemini vectors cannot be cast to 768-dim Groq
nomic-embed vectors, so stored chunks (and their document headers) are
wiped: users re-upload their PDFs once. The new width fits pgvector's
2000-dim HNSW cap, so an ANN index is added for similarity search.

Revision ID: b7e1c4a92f6d
Revises: f3e8f8a12345
Create Date: 2026-11-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b7e1c4a92f6d'
down_revision: Union[str, Sequence[str], None] = 'f3e8f8a12345'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Wipe chunks first (documents.document_id FKs into document_files),
    # then the now-orphaned headers so the UI does not list empty documents.
    op.execute("DELETE FROM documents;")
    op.execute("DELETE FROM document_files;")
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw "
        "ON documents USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_embedding_hnsw;")
    op.execute("DELETE FROM documents;")
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(3072);")
