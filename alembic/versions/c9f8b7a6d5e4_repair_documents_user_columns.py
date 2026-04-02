"""repair missing document ownership schema pieces

Revision ID: c9f8b7a6d5e4
Revises: 8b3a1f2d4c5e
Create Date: 2026-04-02 23:15:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "c9f8b7a6d5e4"
down_revision: Union[str, Sequence[str], None] = "8b3a1f2d4c5e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_indexes(inspector, table_name: str) -> set[str]:
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def _get_fk_names(inspector, table_name: str) -> set[str]:
    names: set[str] = set()
    for fk in inspector.get_foreign_keys(table_name):
        name = fk.get("name")
        if name:
            names.add(name)
    return names


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    if "document_files" not in tables:
        op.create_table(
            "document_files",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("filename", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = inspect(bind)

    doc_file_indexes = _get_indexes(inspector, "document_files")
    if "ix_document_files_user_id" not in doc_file_indexes:
        op.create_index(
            op.f("ix_document_files_user_id"),
            "document_files",
            ["user_id"],
            unique=False,
        )

    if "documents" not in tables:
        return

    document_columns = {col["name"] for col in inspector.get_columns("documents")}
    if "user_id" not in document_columns:
        op.add_column("documents", sa.Column("user_id", sa.UUID(), nullable=True))
    if "document_id" not in document_columns:
        op.add_column("documents", sa.Column("document_id", sa.UUID(), nullable=True))

    inspector = inspect(bind)
    document_fk_names = _get_fk_names(inspector, "documents")
    if "fk_documents_user_id_users" not in document_fk_names:
        op.create_foreign_key(
            "fk_documents_user_id_users",
            "documents",
            "users",
            ["user_id"],
            ["id"],
        )
    if "fk_documents_document_id_document_files" not in document_fk_names:
        op.create_foreign_key(
            "fk_documents_document_id_document_files",
            "documents",
            "document_files",
            ["document_id"],
            ["id"],
        )

    document_indexes = _get_indexes(inspector, "documents")
    if "ix_documents_user_id" not in document_indexes:
        op.create_index(
            op.f("ix_documents_user_id"),
            "documents",
            ["user_id"],
            unique=False,
        )
    if "ix_documents_document_id" not in document_indexes:
        op.create_index(
            op.f("ix_documents_document_id"),
            "documents",
            ["document_id"],
            unique=False,
        )


def downgrade() -> None:
    # Repair migration: keep downgrade intentionally empty.
    pass
