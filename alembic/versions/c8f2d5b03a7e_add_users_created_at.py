"""add users.created_at for guest-account cleanup

Revision ID: c8f2d5b03a7e
Revises: b7e1c4a92f6d
Create Date: 2026-11-19 00:00:01.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c8f2d5b03a7e'
down_revision: Union[str, Sequence[str], None] = 'b7e1c4a92f6d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "created_at")
