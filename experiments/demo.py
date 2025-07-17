import qutip as qt
import qutipext as qte
import numpy as np

from tensoresprit.Experiment import Experiment
from tensoresprit.Extractor import Extractor

import nm_dynamics_learning as nml

import pickle

N = 1
delta = 0.02
no_samples = 1
t_pulse = 100.

dir = "experiments/results/"
filename = f"N_{N}_t_pulse_{t_pulse}_delta_{delta}_no_samples_{no_samples}"
filetype = ".pickle"

times = delta * np.arange(no_samples)

dim = 2**(2*N)

Ls = []

for i in range(dim):
    H_const = nml.hamiltonian_const(N,
        np.sqrt(i) * np.concatenate(([1.],np.array([0 for i in range(N-1)]))))
    Ls.append(qt.liouvillian(H_const))

H_const = nml.hamiltonian_const(N, None)

rho_0 = nml.bistring_op([0 for i in range(N)])
obs = qte.PauliString([3 * i == 0 for i in range(N)])

print(nml.check_nonlin(Ls, rho_0, obs, t_pulse, return_eval=True, tol=1e-3))

measured_data = nml.experiment(N, no_samples, delta, t_pulse, [], H_const=H_const, sim_steps=1000)

with open(dir+filename+filetype, 'wb') as file:
    pickle.dump(measured_data, file)
    pickle.dump(H_const, file)

data = Experiment(measured_data, no_samples, delta)

solverParams = {'mode': 'oscillation'}
extractor = Extractor(data, solver="tensorESPRIT", solverParams=solverParams)
extractor.extract()

true_spectrum, _ = np.linalg.eig(-1j * qt.liouvillian(H_const).full())
index_true = np.argsort(true_spectrum.real)
true_spectrum = true_spectrum[index_true]
recovered_spectrum, _ = np.linalg.eig(extractor.noisyGeneratorEstimate())
index_rec = np.argsort(recovered_spectrum.real)
recovered_spectrum = recovered_spectrum[index_rec]

print(np.linalg.eig(measured_data[0, :, :]))

print(np.linalg.norm(true_spectrum - recovered_spectrum) /
      np.linalg.norm(true_spectrum))
print(true_spectrum)
print(recovered_spectrum.real)
