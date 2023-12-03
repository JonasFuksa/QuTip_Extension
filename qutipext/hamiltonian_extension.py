from .utils import PAULIS, n, bitstring_state, X,Y,Z
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


######## RYDBERG HAMILTONIAN ########
class Lattice:
    def __init__(self, name, dimension, L, unit_vector):
        """
        Initialize a Lattice object.
        
        :param name: The name of the lattice.
        :param dimension: The dimension of the lattice.
        :param L: A parameter associated with the lattice, such as length or size.
        :param unit_vector: A unit vector that defines the lattice. For example, for a 1D lattice, this would be the lattice spacing, i.e. np.array([scalar])
        for a 2D lattice, this would be np.array([[ax,ay],[bx,by]])
        """
        self.name = name
        self.dimension = dimension
        self.L = L
        self.unit_vector = unit_vector
        self.positions = self.generate_positions()

    def __str__(self):
        """
        String representation of the Lattice object.
        """
        return f"Lattice(name={self.name}, dimension={self.dimension}, L={self.L})"
    def generate_positions(self):
        """
        Generate the positions of the lattice sites.
        """
        if self.dimension == 1:
            return np.arange(self.L)*self.unit_vector[0]
        elif self.dimension == 2:
            return np.array([(x*self.unit_vector[0,:]+y*self.unit_vector[1,:]) for x in range(self.L[0]) for y in range(self.L[1])])
        else:
            raise ValueError("Dimension not supported.")
        
def plot_positions(lattice):
    """
    Plot balls given their coordinates in a 2D numpy array.

    Parameters:
    lattice (Lattice object)
    """
    A = lattice.positions
    x = A[:, 0]
    y = A[:, 1]

    plt.scatter(x, y, c='blue', s=100)  # s is the size of each point
    plt.title("Atom Coordinates Plot")
    plt.xlabel(r"X Coordinate ($\mu m$)")
    plt.ylabel(r"Y Coordinate ($\mu m$)")
    plt.grid(True)
    plt.show()

def Vij(r1,r2,C = 862690*2*np.pi):
    """
    Returns the interaction strength between two atoms separated by a distance r1-r2.
    """
    return C/(np.linalg.norm(r1-r2)**6)
def Interaction(lattice):
    """
    Returns the interaction matrix for a given lattice.
    """
    Ham = 0
    for i in range(1,lattice.positions.shape[0]):
        for j in range(i):
            Ham += Vij(lattice.positions[i],lattice.positions[j])*n(i,lattice.positions.shape[0])*n(j,lattice.positions.shape[0])
    return Ham

def Global_Detuning(lattice):
    """
    Returns the global detuning term: sum_i n_i
    """
    return sum([n(i,lattice.positions.shape[0]) for i in range(lattice.positions.shape[0])])

def Global_X(lattice):
    """
    Returns the global X term: sum_i sigma_x_i/2
    """
    return sum([X(i,lattice.positions.shape[0]) for i in range(lattice.positions.shape[0])])/2

def Global_Y(lattice):
    """
    Returns the global Y term: sum_i sigma_y_i/2
    """
    return sum([Y(i,lattice.positions.shape[0]) for i in range(lattice.positions.shape[0])])/2