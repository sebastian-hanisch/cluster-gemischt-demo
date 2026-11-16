"""Gemeinsame Fixtures und Hilfen: Handbeispiele in Brüchen (DEMO-PLAYBOOK §1b), Bruch-Rechner für kleine Matrizen."""

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAND1 = [[1, 3], [5, 9], [8, 10]]
HAND2 = [[1, 3], [2, 6], [4, 4]]


@pytest.fixture
def hand1():
    """Drei Kunden mit je zwei Containern: Kundenmittel 2, 7, 9, Gesamtmittel 6, SSB 52, SSW 12 (von Hand)."""
    return HAND1


@pytest.fixture
def hand2():
    """Randfall: Kundenmittel 2, 4, 4 und Gesamtmittel 10/3; der ANOVA-Schätzer wird negativ."""
    return HAND2


def bruch_matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def bruch_transp(A):
    return [list(r) for r in zip(*A)]


def bruch_inv(A):
    """Gauß-Jordan in Brüchen (kleine Matrizen)."""
    n = len(A)
    M = [[F(v) for v in A[i]] + [F(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def bruch_V(g, s2, su2):
    """Dichte Kovarianz V = σ²I + σ_u²ZZ' in Brüchen."""
    n = len(g)
    return [[(s2 if i == j else F(0)) + (su2 if g[i] == g[j] else F(0)) for j in range(n)] for i in range(n)]


def dichtes_V(g, s2, su2):
    g = np.asarray(g)
    return s2 * np.eye(len(g)) + su2 * (g[:, None] == g[None, :])
