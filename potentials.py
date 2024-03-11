import numpy as np


def harmonic_oscillator(x, omega=1):
    return 0.5 * omega**2 * x**2


def double_well(x, omega=1, a=1):
    """
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.59.2070
    """
    return 0.5 * omega**2 / (4 * a**2) * (x**2 - a**2) ** 2


def Coulomb(x, Z=1, R=0, a=1):
    return -Z / np.sqrt((x - R) ** 2 + a)
