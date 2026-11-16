"""Ganzzahliger Zufallsgenerator (SplitMix64): gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen.

Herkunft: `SplitMix64`, `exact_log` und `exact_cos` sind eine Kopie der geprüften Vorlage `tools/templates/rng_splitmix64_exact.py` des Website-Repos
(Normalzahl über `decimal`, 45 Stellen, einmal auf double gerundet), nur die Namen der Konstanten sind angepasst. Die Feld-Fassung (`splitmix_array`,
`uniform_array`, `log_array`, `cos_2pi_array`, `normal_array`) ist hier neu nach dem Muster der Regelungs-Demos geschrieben: gleiche Folge wie die Klasse,
vektorisiert, und Logarithmus und Kosinus **nur aus Grundrechenarten** (Reihen; IEEE 754 rundet +, -, *, / und die Wurzel exakt, es gibt keinen Anteil der
C-Bibliothek). Der Grund: die exakte Normalzahl der Klasse kostet etwa 80 bis 140 µs, die Messreihen brauchen je 1000 Wiederholungen bis zu 15 Millionen Werte.
Die Gleichheit mit der exakten Fassung (Abweichung höchstens 1e-13) prüft `tests/test_rng.py`.
Python-`hash()`, `random`, `math.log` und `math.cos` sind für feste Zufalls-Seeds über Plattformen nicht geeignet; `numpy.random` ebenso."""

import decimal
import math

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15

_CTX = decimal.Context(prec=45)


def exact_log(x: float) -> float:
    """Natürlicher Logarithmus, für x > 0, auf jeder Plattform bitgleich (praktisch auf das nächste double gerundet)."""
    return float(_CTX.ln(decimal.Decimal(x)))


def exact_cos(x: float) -> float:
    """Kosinus für |x| bis etwa 10, auf jeder Plattform bitgleich (Taylorreihe in 45 Stellen, dann einmal auf double gerundet)."""
    d = decimal.Decimal(x)
    x2 = _CTX.multiply(d, d)
    term = decimal.Decimal(1)
    total = decimal.Decimal(1)
    k = 0
    limit = decimal.Decimal(10) ** -48
    while True:
        k += 2
        term = _CTX.divide(_CTX.multiply(term, x2), k * (k - 1))
        total = _CTX.subtract(total, term) if (k // 2) % 2 == 1 else _CTX.add(total, term)
        if term < limit:
            break
    return float(total)


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def below(self, n: int) -> int:
        """Ganzzahl in 0..n-1 (Rest der Division; der Fehler 2^-64 · n ist hier ohne Belang)."""
        return self.next_u64() % n

    def normal(self) -> float:
        """Standardnormalverteilt (Box-Muller, nur der Kosinus-Zweig)."""
        u1 = max(self.random(), 1e-12)
        u2 = self.random()
        return math.sqrt(-2.0 * exact_log(u1)) * exact_cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Feld-Fassung (vektorisiert)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def splitmix_array(seed: int, n: int) -> np.ndarray:
    """Die ersten n Ausgaben von SplitMix64(seed) als uint64-Feld (gleiche Folge wie die Klasse)."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & MASK) + k * np.uint64(GAMMA)          # uint64-Felder rechnen modulo 2^64
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, n: int) -> np.ndarray:
    """n Gleichverteilte in [0, 1) mit 53 Bit (gleiche Werte wie `SplitMix64.random`)."""
    return (splitmix_array(seed, n) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


_LN2_HI = 6.93147180369123816490e-01     # ln 2 in zwei Teilen; hi hat wenige Mantissenbits, e * hi ist für |e| < 2^11 exakt
_LN2_LO = 1.90821492927058770002e-10
_TWO_PI = 2.0 * math.pi


def log_array(x: np.ndarray) -> np.ndarray:
    """Natürlicher Logarithmus für x > 0: x = m 2^e mit m in [0,75; 1,5), ln m = 2 artanh((m-1)/(m+1)) als Reihe, dazu e ln 2.
    Nur +, -, *, / und frexp (exakt): auf jeder Plattform dieselben Bits."""
    m, e = np.frexp(x)
    klein = m < 0.75
    m = np.where(klein, m * 2.0, m)
    e = np.where(klein, e - 1, e).astype(np.float64)
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    acc = np.zeros_like(s)
    for k in range(25, 0, -2):                     # Horner über 1/k s^(k-1); |s| <= 0,2, der Rest liegt unter 1e-19
        acc = acc * s2 + 1.0 / k
    return 2.0 * s * acc + (e * _LN2_HI + e * _LN2_LO)


def cos_2pi_array(u: np.ndarray) -> np.ndarray:
    """cos(2 pi u) für u in [0, 1): auf [0; 1/8] gespiegelt, dort Taylorreihe von cos bzw. sin (Winkel höchstens pi/4)."""
    a = np.where(u > 0.5, 1.0 - u, u)
    neg = a > 0.25
    a = np.where(neg, 0.5 - a, a)
    use_sin = a > 0.125
    b = np.where(use_sin, 0.25 - a, a)
    x = _TWO_PI * b
    x2 = x * x
    c = np.zeros_like(x)
    for k in range(24, 0, -2):
        c = 1.0 - c * x2 / ((k - 1) * k)
    s = np.zeros_like(x)
    for k in range(25, 1, -2):
        s = 1.0 - s * x2 / ((k - 1) * k)
    s = s * x
    val = np.where(use_sin, s, c)
    return np.where(neg, -val, val)


def normal_from_uniform(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Standardnormalwerte aus je zwei Gleichverteilten (Box-Muller, Kosinus-Zweig; u1 wird bei 1e-12 abgeschnitten)."""
    return np.sqrt(-2.0 * log_array(np.maximum(u1, 1e-12))) * cos_2pi_array(u2)


def normal_array(seed: int, n: int) -> np.ndarray:
    """n Standardnormalwerte; Wert j nutzt die Ausgaben 2j und 2j + 1, wie `SplitMix64.normal`."""
    u = uniform_array(seed, 2 * n)
    return normal_from_uniform(u[0::2], u[1::2])
