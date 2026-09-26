"""add agent_outputs mechanical flags (stale_data, anomalies, data_coverage)

Revision ID: c1a4e9f2b7d3
Revises: 416d5dbac42d
Create Date: 2026-09-25 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1a4e9f2b7d3'
down_revision: Union[str, None] = '416d5dbac42d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_outputs', sa.Column('stale_data', sa.JSON(), nullable=True))
    op.add_column('agent_outputs', sa.Column('anomalies', sa.JSON(), nullable=True))
    op.add_column('agent_outputs', sa.Column('data_coverage', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('agent_outputs', 'data_coverage')
    op.drop_column('agent_outputs', 'anomalies')
    op.drop_column('agent_outputs', 'stale_data')
