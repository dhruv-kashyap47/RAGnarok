"""update embedding dimensions

Revision ID: f3e8f8a12345
Revises: c9f8b7a6d5e4
Create Date: 2026-04-05 17:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3e8f8a12345'
down_revision: Union[str, Sequence[str], None] = 'c9f8b7a6d5e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use raw SQL to alter the vector column dimension
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(3072);")


def downgrade() -> None:
    # Revert to 768 dimensions
    op.execute("ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768);")
