"""Pydantic schemas for the Analysis API (86bbuhjup), matching
docs/api-conventions.md's "Analysis Workflow" section and Error Codes table.
"""
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnalysisCreate(BaseModel):
    ticker: str
    account_type: Literal["tfsa", "rrsp", "trading"]
    timeline: Literal["short_term", "medium_term", "long_term"]
    # Required, not optional: AnalysisRun.user_id is a NOT NULL FK, and MVP
    # has no auth (docs/api-conventions.md's own "Authentication" section --
    # user identity is just a client-held UUID, no session to derive it
    # from server-side).
    user_id: UUID


class AnalysisCreateResponse(BaseModel):
    analysis_id: UUID
    status: str


class AnalysisProgress(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pass_: int = Field(serialization_alias="pass")
    agents_complete: int


class AnalysisStatusResponse(BaseModel):
    status: str
    progress: AnalysisProgress | None = None


class AgentOutputSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_name: str
    agent_pass: str
    status: str
    recommendation: str | None
    confidence: int | None
    analysis_confidence: str | None
    narrative: str | None
    structured_output: dict | None


class AnalysisResultResponse(BaseModel):
    analysis_id: UUID
    status: str
    stock_outlook_direction: str | None = None
    overall_confidence: int | None = None
    account_recommendation: dict | None = None
    position_size_suggestion: str | None = None
    expected_return_tier: str | None = None
    key_drivers: list | None = None
    synthesis_narrative: str | None = None
    bull_case_strength: int | None = None
    bear_case_strength: int | None = None
    disagreement_score: int | None = None
    disagreement_class: str | None = None
    agent_outputs: list[AgentOutputSummary] = []
