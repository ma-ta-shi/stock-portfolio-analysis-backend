"""user_profiles: add tax context, TFSA/RRSP room, time-sensitive cash needs,
and Portfolio Optimizer input fields; drop the unused financial_profile and
context_and_goals JSON columns (86bc8efe7)

Revision ID: fb966212df72
Revises: 46c9db910552
Create Date: 2026-09-28 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fb966212df72'
down_revision: Union[str, None] = '46c9db910552'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.drop_column('financial_profile')
        batch_op.drop_column('context_and_goals')
        batch_op.add_column(sa.Column('province', sa.String(length=2), nullable=True))
        batch_op.add_column(sa.Column('marginal_tax_rate_override_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('income_annual', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('tfsa_room_remaining', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('tfsa_room_as_of', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('rrsp_room_remaining', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('rrsp_room_as_of', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('time_sensitive_cash_needs', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('max_position_size_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('max_sector_weight_pct', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('rrsp_annual_contribution_cap', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.drop_column('rrsp_annual_contribution_cap')
        batch_op.drop_column('max_sector_weight_pct')
        batch_op.drop_column('max_position_size_pct')
        batch_op.drop_column('time_sensitive_cash_needs')
        batch_op.drop_column('rrsp_room_as_of')
        batch_op.drop_column('rrsp_room_remaining')
        batch_op.drop_column('tfsa_room_as_of')
        batch_op.drop_column('tfsa_room_remaining')
        batch_op.drop_column('income_annual')
        batch_op.drop_column('marginal_tax_rate_override_pct')
        batch_op.drop_column('province')
        batch_op.add_column(sa.Column('financial_profile', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('context_and_goals', sa.JSON(), nullable=True))
