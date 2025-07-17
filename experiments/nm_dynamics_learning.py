import qutip as qt
import qutipext as qte
import numpy as np
import scipy.linalg as sla


def lattice_setup(N):
    return qte.Lattice("Chain", 1, N, np.array([[8.7, 0]]))


def hamiltonian_const(N, mu):
    lattice = lattice_setup(N)

    if mu is None:
        print("Local detuning not provided. Drawing a random one.")
        mu = np.random.rand(N)

    V = qte.Interaction(lattice)
    Detune = -(2*np.pi*2)*qte.Global_Detuning(lattice)
    Rabi = (2*np.pi*2)*qte.Global_X(lattice)
    Local_Detune = qte.Local_Detuning(lattice, mu)

    return V + Detune + Rabi + Local_Detune


def pulse(N, index, t, t_pulse):
    lattice = lattice_setup(N)

    V = qte.Interaction(lattice)
    Detune = -(2*np.pi*2)*qte.Global_Detuning(lattice)
    Rabi = (2*np.pi*2)*qte.Global_X(lattice)
    mu = np.concatenate(([1], np.array([0 for i in range(N-1)], dtype=int)))
    Local_Detune = np.sqrt(index) * qte.Local_Detuning(lattice, mu)

    #mu = np.concatenate(([100 * np.sin(2 * np.pi * index * t / t_pulse)],
    #                     np.array([0 for i in range(N-1)])))
    #mu = np.concatenate(([100 * np.sqrt(index)], np.array([0 for i in range(N-1)])))

    return V + Detune + Rabi + Local_Detune


def hamiltonian(N, i, j, t, t_free, t_pulse, H_const):
    if t < t_pulse:
        return pulse(N, i, t, t_pulse)
    elif t < t_pulse + t_free:
        return H_const
    else:
        return pulse(N, j, t, t_pulse)


def _experiment(t_free, N, i, j, t_pulse, H_const, rho_0, obs, sim_steps,
                c_ops):

    H_current = qt.QobjEvo(lambda t: hamiltonian(N, i, j, t,
                                                 t_free,
                                                 t_pulse,
                                                 H_const))
    total_time = 2*t_pulse + t_free
    solver_times = np.linspace(0, total_time, sim_steps)
    return qt.sesolve(H_current, rho_0, solver_times,
                      e_ops=obs).expect[0][-1]


def experiment(N, no_samples, delta, t_pulse, c_ops, H_const=None,
               sim_steps=100):
    if H_const is None:
        H_const = hamiltonian_const(N, None)

    dim = 2**(2*N)
    times = delta * np.arange(no_samples)

    measured_data = np.zeros((no_samples, dim, dim), dtype=complex)

    rho_0 = bistring_op([0 for i in range(N)])
    obs = qte.PauliString(np.concatenate(([3],
                                          np.array([0 for i in range(N-1)],
                                                     dtype=int))))

    for i in range(dim):
        for j in range(dim):
            print("working on indices", i, j)
            if j == dim - 1:
                curr_obs = qte.PauliString([0 for i in range(N)])
            else:
                curr_obs = obs

            #res = [_current_experiment(t) for t in times]
            measured_data[:, i, j] = qt.parallel_map(_experiment,
                                                     times,
                                                     task_args=(N, i, j, t_pulse,
                                                      H_const, rho_0, curr_obs,
                                                      sim_steps, c_ops),
                                                     progress_bar=True)
    return measured_data


def check_nonlin(Ls, rho_0, obs, t_pulse, tol=1e-10, return_eval=False):
    rho_0_vec = qt.operator_to_vector(rho_0).full()
    obs_vec = qt.operator_to_vector(obs).full()

    dim = len(rho_0_vec)
    N = int(np.log2(np.sqrt(dim)))

    obs_id_vec = qt.operator_to_vector(qte.PauliString([0 for i in range(N)])).full()

    mat = np.zeros((dim, dim), dtype=complex)

    for i, L_init in enumerate(Ls):
        for j, L_fin in enumerate(Ls):
            init_vec = sla.expm(t_pulse * L_init.full()) @ rho_0_vec
            if j == dim-1:
                fin_vec = obs_id_vec.conj().T @ sla.expm(t_pulse * L_fin.full())
            else:
                fin_vec = obs_vec.conj().T @ sla.expm(t_pulse * L_fin.full())
            mat[i, j] = fin_vec @ init_vec
    evals, _ = np.linalg.eig(mat)

    magnitudes = np.sort(np.abs(evals))

    smallest_eval = magnitudes[0]

    if return_eval:
        return smallest_eval
    else:
        return smallest_eval > tol


def bistring_op(bitstring):
    vec = qte.bitstring_state(bitstring)
    return vec * vec.dag()
