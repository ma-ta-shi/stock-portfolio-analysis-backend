"""add agent_outputs data_quality_assessment (D6 section 3 per-agent rollup)

Revision ID: d4e2f9a1c6b8
Revises: c1a4e9f2b7d3
Create Date: 2026-09-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e2f9a1c6b8'
down_revision: Union[str, None] = 'c1a4e9f2b7d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_outputs', sa.Column('data_quality_assessment', sa.String(length=10), nullable=True))


def downgrade() -> None:
    op.drop_column('agent_outputs', 'data_quality_assessment')
