import numpy as np


class sine_laser:
    def __init__(self, E0, omega, td):
        self.E0 = E0
        self.omega = omega
        self.td = td

    def __call__(self, t):
        if t <= self.td:
            return self.E0 * np.sin(self.omega * t)
        else:
            return 0.0


class sine_square_laser:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.F_str = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.F_str
        )
        return pulse
