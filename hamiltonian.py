import numpy as np
from scipy.integrate import simps


class Hamiltonian:
    def __init__(self, x, potential, electric_field=None):
        self.x = x
        self.n_x = len(x)
        self.dx = x[1] - x[0]

        self.T = (
            np.diag(-1 / self.dx**2 * np.ones(self.n_x - 1), k=-1)
            + np.diag(2 / self.dx**2 * np.ones(self.n_x))
            + np.diag(-1 / self.dx**2 * np.ones(self.n_x - 1), k=1)
        )
        self.T *= 0.5

        self.V = np.diag(potential)

        self.H = self.T + self.V
        self.electric_field = electric_field

    def set_electric_field(self, electric_field):
        self.electric_field = electric_field

    def get_eigenstates(self, n_eigenstates=5):

        eps, C = np.linalg.eigh(self.H)
        psi = np.zeros((self.n_x, n_eigenstates))

        # eigh returns eigenstates that are orthonormal with respect to the vector product.
        # Normalize the eigenstates with respect to the integral over space.
        for i in range(n_eigenstates):
            psi[:, i] = C[:, i] / np.sqrt(simps(C[:, i] ** 2, self.x))
        return eps[:n_eigenstates], psi

    def __call__(self, t):
        if self.electric_field is not None:
            return self.H + self.electric_field(t) * np.diag(self.x)
        else:
            return self.H
