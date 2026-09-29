"""Fourier Learning Machine (FLM): an m-dimensional PyTorch implementation.

The FLM is a feedforward network whose parameters correspond directly to the
quantities of a multidimensional nonharmonic Fourier series in cosine
phase-shifted form. It is built from N sub-networks; each sub-network holds one
learnable frequency vector n and l = 2**(m-1) cosine-activated hidden neurons:

    H_n(x) = sum_i A_i * cos((e^(i) . n) . x - phi_i)

where e^(i) is the i-th row of the fixed m-Lexi Sign Matrix S^(m). The full FLM
output is the ordinary sum over sub-networks: f(x) = sum_n H_n(x).

Basic use:

    from flm import FourierLearningMachine, to_reference_domain
    import torch

    model = FourierLearningMachine(m=2, N=16, seed=0)
    x = torch.zeros(2)            # a single point in [-pi, pi]^2
    y = model(x)                  # scalar output

    X = torch.rand(100, 2) * 2 * torch.pi - torch.pi   # a batch of points
    Y = torch.vmap(model)(X)      # shape (100,)

See the accompanying notebook for a worked training demo.
"""

import math
from itertools import product

import torch
import torch.nn as nn

__all__ = [
    "lexi_sign_matrix",
    "lattice_frequencies",
    "FourierLearningMachine",
    "to_reference_domain",
]


def lexi_sign_matrix(m: int) -> torch.Tensor:
    """Return S^(m): shape (2**(m-1), m). First column +1; the rest over {+1,-1}^(m-1)."""
    tail = list(product([1, -1], repeat=m - 1))   # yields +1 before -1 -> lexicographic
    rows = [(1,) + t for t in tail]
    return torch.tensor(rows, dtype=torch.get_default_dtype())


def lattice_frequencies(N: int, m: int) -> torch.Tensor:
    """First N points of N_0^m ordered by (Euclidean norm, then lexicographically). Shape (N, m)."""
    K = 1
    while True:
        pts = list(product(range(K + 1), repeat=m))          # {0,...,K}^m
        # points with norm <= K are guaranteed complete (anything excluded has a coord > K)
        pts = [p for p in pts if sum(c * c for c in p) <= K * K]
        if len(pts) >= N:
            pts.sort(key=lambda p: (sum(c * c for c in p), p))
            return torch.tensor(pts[:N], dtype=torch.get_default_dtype())
        K += 1


class FourierLearningMachine(nn.Module):
    """Arguments:
        m: input dimension.
        N: network size = number of sub-networks (frequency vectors).
        seed: optional int for reproducible phase initialisation.

    NN Input:  x of shape (m,), a single point assumed scaled to [-pi, pi]^m.
    NN Output: scalar f(x).

    To evaluate on a batch X of shape (B, m), use torch.vmap(model)(X) -> shape (B,).
    """

    def __init__(self, m: int, N: int, seed: int | None = None):
        super().__init__()
        self.m = m
        self.N = N
        self.l = 2 ** (m - 1)     # hidden neurons per sub-network

        if seed is not None:
            torch.manual_seed(seed)

        # Fixed sign structure S^(m) (NOT trainable)
        self.register_buffer("sign_matrix", lexi_sign_matrix(m))     # size --> (l, m)

        # Trainable parameters (initialization)
        self.frequencies = nn.Parameter(lattice_frequencies(N, m))            # size --> (N, m)
        self.phases      = nn.Parameter(torch.randn(N, self.l) * (math.pi / 3))  # size --> (N, l)
        self.amplitudes  = nn.Parameter(torch.zeros(N, self.l))               # size --> (N, l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.frequencies.unsqueeze(1) * self.sign_matrix     # size --> (N, l, m) | e^(i) ⊙ n
        z = W @ x - self.phases                                  # size --> (N, l)    | (e^(i) ⊙ n)·x - phi
        H = (self.amplitudes * torch.cos(z)).sum(dim=1)          # size --> (N,)      | H_n = sum_i A_i cos(...)
        return H.sum()                                           # scalar             | f = sum_n H_n


def to_reference_domain(x: torch.Tensor, lows, highs) -> torch.Tensor:
    """Affinely map x from the box [lows, highs] (per dimension) onto [-pi, pi]^m."""
    lows  = torch.as_tensor(lows,  dtype=x.dtype, device=x.device)
    highs = torch.as_tensor(highs, dtype=x.dtype, device=x.device)
    return (x - lows) / (highs - lows) * (2 * math.pi) - math.pi
