import numpy as np
import qutip as qt

class ReadOnlyList:
    def __init__(self, iterable):
        self._internal_list = list(iterable)

    def __getitem__(self, key):
        return self._internal_list[key]

    def __len__(self):
        return len(self._internal_list)

    def __iter__(self):
        return iter(self._internal_list)

    # You can add more methods as needed, but avoid adding methods that modify the list

PAULIS = ReadOnlyList([qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()])

# Create a cache for identity matrices to avoid recreation
identity_cache = {}

def get_identity(dim):
    """Retrieve or create a cached identity matrix."""
    if dim not in identity_cache:
        identity_cache[dim] = qt.qeye(dim)
    return identity_cache[dim]

def n(i, qubits):
    # Use a list comprehension and built-in QuTiP functions
    lst = [get_identity(2) if k != i else qt.basis(2, 1) * qt.basis(2, 1).dag() for k in range(qubits)]
    return qt.tensor(lst)
def X(i, qubits):
    # Use a list comprehension and built-in QuTiP functions
    lst = [get_identity(2) if k != i else qt.sigmax() for k in range(qubits)]
    return qt.tensor(lst)
def Y(i, qubits):
    # Use a list comprehension and built-in QuTiP functions
    lst = [get_identity(2) if k != i else qt.sigmay() for k in range(qubits)]
    return qt.tensor(lst)
def Z(i, qubits):
    # Use a list comprehension and built-in QuTiP functions
    lst = [get_identity(2) if k != i else qt.sigmaz() for k in range(qubits)]
    return qt.tensor(lst)

singleZ_cache = {}

def get_singleZ(i,N):
    """Retrieve or create a cached identity matrix."""
    if (i,N) not in singleZ_cache:
        singleZ_cache[(i,N)] = Z(i,N)
    return singleZ_cache[(i,N)]

singleX_cache = {}
def get_singleX(i,N):
    """Retrieve or create a cached identity matrix."""
    if (i,N) not in singleX_cache:
        singleX_cache[(i,N)] = X(i,N)
    return singleX_cache[(i,N)]

singleY_cache = {}
def get_singleY(i,N):
    """Retrieve or create a cached identity matrix."""
    if (i,N) not in singleY_cache:
        singleY_cache[(i,N)] = Y(i,N)
    return singleY_cache[(i,N)]

def bitstring_state(b):
    """
    Returns the computational basis state corresponding to the bitstring b.
    """
    return qt.tensor([qt.basis(2,int(x)) for x in b])

def PauliString(list_of_pauli):
    """
    Returns the PauliString corresponding to the list of Pauli matrices.
    Input: list of Pauli matrices, i.e. [0,2,3,0]==IYZI
    """
    return qt.tensor([PAULIS[x] for x in list_of_pauli])