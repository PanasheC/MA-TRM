"""Interfaces for differentiable latent communication links."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class RecursiveLink(nn.Module, ABC):
    """Base class for a latent message transformation."""

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
