import numpy as np
import pickle
from tensoresprit.Experiment import Experiment
from tensoresprit.Extractor import Extractor
import qutip as qt

N = 1
t_pulse = 100.
delta = 0.02
no_samples = 25
t_pulse = 100.

dir = "experiments/results/"
filename = f"N_{N}_t_pulse_{t_pulse}_delta_{delta}_no_samples_{no_samples}_t_pulse_{t_pulse}"
filetype = ".pickle"

with open (dir+filename+filetype, 'rb') as file:
    measured_data = pickle.load(file)
    H_const = pickle.load(file)

print(measured_data[0,:,:])
data = Experiment(measured_data, no_samples, delta)

solverParams = {'mode': 'oscillation'}
extractor = Extractor(data, solver="tensorESPRIT", solverParams=solverParams)
extractor.extract()

true_spectrum, _ = np.linalg.eig(-1j*qt.liouvillian(H_const).full())
index_true = np.argsort(true_spectrum.real)
true_spectrum = true_spectrum[index_true]
recovered_spectrum, _ = np.linalg.eig(extractor.noisyGeneratorEstimate())
index_rec = np.argsort(recovered_spectrum.real)
recovered_spectrum = recovered_spectrum[index_rec]
print(extractor.noisyGeneratorEstimate())

print(np.linalg.norm(true_spectrum - recovered_spectrum) /
      np.linalg.norm(true_spectrum))
print(true_spectrum)
print(recovered_spectrum.real)
