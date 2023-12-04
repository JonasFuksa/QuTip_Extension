from .utils import PAULIS, n, bitstring_state, X,Y,Z
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


######## RYDBERG HAMILTONIAN ########
def generate_random_y(A, sample_size,C = 862690*2*np.pi):
    # Generate uniformly distributed y values between -A and A
    y_uniform = np.random.uniform(-A/C, A/C, size = sample_size)

    # We need to transform these y values to get the corresponding x values
    # Since y = 1/x^6, we solve for x: x = (1/y)^(1/6)
    # We handle positive and negative y values separately to maintain the sign of x
    x_transformed = np.sign(y_uniform) * np.abs(1 / y_uniform)**(1/6)

    # Recalculate y to verify the distribution
    # y_transformed = np.sign(y_uniform)*1 / x_transformed**6

    return x_transformed

class Lattice:
    def __init__(self, name, dimension, L, unit_vector, random_shift = False, shift_sigma = 0.1):
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
        self.random_shift = random_shift
        self.shift_sigma = shift_sigma
        self.positions = self.generate_positions()

    def __str__(self):
        """
        String representation of the Lattice object.
        """
        return f"Lattice(name={self.name}, dimension={self.dimension}, L={self.L})"
    def update_positions(self,positions):
        """
        Update the positions of the lattice sites.
        """
        self.positions = positions
    def generate_positions(self):
        """
        Generate the positions of the lattice sites.
        """
        if self.dimension == 1:
            if self.random_shift:
                base = np.array([(x*self.unit_vector[0,:]) for x in range(self.L)])
                shift = generate_random_y(self.shift_sigma,base.shape)
                # shift = np.random.uniform(-self.shift_sigma,self.shift_sigma,size = base.shape)
                return base+shift
            else:
                return np.array([(x*self.unit_vector[0,:]) for x in range(self.L)])
            # return np.arange(self.L)*self.unit_vector[0,:]
        elif self.dimension == 2:
            if self.random_shift:
                base = np.array([(x*self.unit_vector[0,:]+y*self.unit_vector[1,:]) for x in range(self.L[0]) for y in range(self.L[1])])
                shift = generate_random_y(self.shift_sigma,base.shape)
                # shift = np.random.uniform(-self.shift_sigma,self.shift_sigma,size = base.shape)
                return base+shift
            else:
                return np.array([(x*self.unit_vector[0,:]+y*self.unit_vector[1,:]) for x in range(self.L[0]) for y in range(self.L[1])])
        else:
            raise ValueError("Dimension not supported.")
        
def plot_positions(lattice):
    """
    Plot balls given their coordinates in a 2D numpy array.

    Parameters:
    lattice (Lattice object)
    """
    if type(lattice) == Lattice:
        A = lattice.positions
    elif type(lattice) == np.ndarray:
        A = lattice
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