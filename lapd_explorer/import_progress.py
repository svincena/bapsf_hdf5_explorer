"""Acquisition counts accompanying fractional import work progress."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShotProgress:
    stage: str
    completed: int
    total: int
    phase: int = 1
    channel: int = 1
    channels: int = 1
