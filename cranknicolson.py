import numpy as np
from scipy.integrate import simps
from scipy.linalg import solve_banded
import tqdm


def run_CN(hamiltonian, psi0, x, dt, tfinal, eigenstates, mask=None):

    psi_t = psi0.copy()
    num_steps = int(tfinal / dt) + 1
    time_points = np.zeros(num_steps)
    n_x = len(x)
    n_eigenstates = eigenstates.shape[1]

    psi_hist = np.zeros((num_steps, n_x), dtype=np.complex128)
    psi_hist[0] = psi_t.copy()
    norm_t = np.zeros(num_steps)
    norm_t[0] = simps(np.abs(psi_t) ** 2, x)
    expec_x = np.zeros(num_steps, dtype=np.complex128)
    expec_x[0] = simps(psi_t.conj() * x * psi_t, x)

    populations = np.zeros((n_eigenstates, num_steps), dtype=np.complex128)
    for i in range(eigenstates.shape[1]):
        populations[i, 0] = simps(psi_t * eigenstates[:, i], x)

    I = np.eye(n_x)
    for n in tqdm.tqdm(range(num_steps - 1)):
        H_t = hamiltonian(time_points[n] + dt / 2)
        A_m = I - 1j * dt / 2 * H_t
        A_p = I + 1j * dt / 2 * H_t
        Ap_diag = np.diag(A_p, k=0)
        Ap_upperdiag = np.zeros(n_x, dtype=np.complex128)
        Ap_upperdiag[1:] = np.diag(A_p, k=1)
        Ap_subdiag = np.zeros(n_x, dtype=np.complex128)
        Ap_subdiag[0:-1] = np.diag(A_p, k=-1)

        z = np.dot(A_m, psi_t)

        psi_t = solve_banded(
            (1, 1),
            np.array([Ap_upperdiag, Ap_diag, Ap_subdiag]),
            z,
        )

        if mask is not None:
            psi_t *= mask
        expec_x[n + 1] = simps(psi_t.conj() * x * psi_t, x)
        psi_hist[n + 1] = psi_t
        norm_t[n + 1] = simps(np.abs(psi_t) ** 2, x)
        for i in range(n_eigenstates):
            populations[i, n + 1] = simps(psi_t * eigenstates[:, i], x)
        time_points[n + 1] = (n + 1) * dt

    data = dict()
    data["time_points"] = time_points
    data["psi_hist"] = psi_hist
    data["expec_x"] = expec_x
    data["populations"] = populations
    data["norm"] = norm_t

    return data
