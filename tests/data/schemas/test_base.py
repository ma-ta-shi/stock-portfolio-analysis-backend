import pytest
from pydantic import ValidationError

from data.schemas.base import ContractModel


class _Sample(ContractModel):
    value: int


def test_contract_model_is_frozen():
    """Immutable snapshot — mutating after construction must raise, not
    silently succeed (these are handed to agents once per run)."""
    instance = _Sample(value=1)
    with pytest.raises(ValidationError):
        instance.value = 2


def test_contract_model_forbids_extra_fields():
    """Catches a precompute module silently adding/renaming a field
    before it reaches an agent prompt."""
    with pytest.raises(ValidationError):
        _Sample(value=1, unexpected_field="surprise")


def test_contract_model_accepts_declared_fields():
    instance = _Sample(value=1)
    assert instance.value == 1
