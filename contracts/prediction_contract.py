"""Stable boundary between Person 1's ML pipeline and Person 2's API/UI."""

from __future__ import annotations

from typing import Any, TypedDict


class RoutePrediction(TypedDict):
    trip_id: str
    arc_ids: list[int]
    candidate_scores: list[dict[str, Any]]
    route_continuity: float | None
    model_version: str


def predict_route(trip_id: str) -> RoutePrediction:
    """Load exported artifacts and return one decoded route.

    Person 1 implements this function behind the contract. Person 2 should
    call it through the API adapter and must not depend on internal model code.
    """

    raise NotImplementedError("Person 1 will implement the model adapter")
