"""Type aliases mirroring the production rLIC package."""

__all__ = [
    "Boundary",
    "UVMode",
    "BoundaryDict",
    "BoundaryPair",
]

from typing import Literal, TypeAlias, TypedDict

Boundary = Literal["closed", "periodic"]
BoundaryPair = tuple[Boundary, Boundary]
AnyBoundary: TypeAlias = Boundary | BoundaryPair


class BoundaryDict(TypedDict):
    x: AnyBoundary
    y: AnyBoundary


UVMode = Literal["velocity", "polarization"]
