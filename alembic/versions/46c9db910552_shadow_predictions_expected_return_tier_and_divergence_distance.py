"""shadow_predictions: rename *_projected_return_tier to *_expected_return_tier,
add primary_cio_outlook_distance and high_divergence (86bbt1kpj)

Revision ID: 46c9db910552
Revises: d4e2f9a1c6b8
Create Date: 2026-09-27 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '46c9db910552'
down_revision: Union[str, None] = 'd4e2f9a1c6b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('shadow_predictions') as batch_op:
        batch_op.alter_column(
            'primary_projected_return_tier',
            new_column_name='primary_expected_return_tier',
            existing_type=sa.String(length=20),
            existing_nullable=True,
        )
        batch_op.alter_column(
            'shadow_projected_return_tier',
            new_column_name='shadow_expected_return_tier',
            existing_type=sa.String(length=20),
            existing_nullable=True,
        )
        batch_op.add_column(
            sa.Column('primary_cio_outlook_distance', sa.Integer(), nullable=True),
            insert_after='divergence_magnitude',
        )
        batch_op.add_column(
            sa.Column('high_divergence', sa.Boolean(), nullable=True),
            insert_after='primary_cio_outlook_distance',
        )


def downgrade() -> None:
    with op.batch_alter_table('shadow_predictions') as batch_op:
        batch_op.drop_column('high_divergence')
        batch_op.drop_column('primary_cio_outlook_distance')
        batch_op.alter_column(
            'shadow_expected_return_tier',
            new_column_name='shadow_projected_return_tier',
            existing_type=sa.String(length=20),
            existing_nullable=True,
        )
        batch_op.alter_column(
            'primary_expected_return_tier',
            new_column_name='primary_projected_return_tier',
            existing_type=sa.String(length=20),
            existing_nullable=True,
        )
