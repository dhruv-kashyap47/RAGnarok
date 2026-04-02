"""add document metadata and user isolation

Revision ID: 8b3a1f2d4c5e
Revises: 4a6f9c1d2b7e
Create Date: 2026-04-01 20:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b3a1f2d4c5e'
down_revision: Union[str, Sequence[str], None] = '4a6f9c1d2b7e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'document_files',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_files_user_id'), 'document_files', ['user_id'], unique=False)

    op.add_column('documents', sa.Column('user_id', sa.UUID(), nullable=True))
    op.add_column('documents', sa.Column('document_id', sa.UUID(), nullable=True))
    op.create_foreign_key('fk_documents_user_id_users', 'documents', 'users', ['user_id'], ['id'])
    op.create_foreign_key('fk_documents_document_id_document_files', 'documents', 'document_files', ['document_id'], ['id'])
    op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
    op.create_index(op.f('ix_documents_document_id'), 'documents', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_documents_document_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_constraint('fk_documents_document_id_document_files', 'documents', type_='foreignkey')
    op.drop_constraint('fk_documents_user_id_users', 'documents', type_='foreignkey')
    op.drop_column('documents', 'document_id')
    op.drop_column('documents', 'user_id')

    op.drop_index(op.f('ix_document_files_user_id'), table_name='document_files')
    op.drop_table('document_files')
