import matplotlib.pyplot as plt
import numpy as np
from ase.io import Trajectory

traj = Trajectory("simulation/trajectory.traj")

energies = [atoms.get_potential_energy() for atoms in traj]

plt.plot(np.arange(len(energies)), energies, marker='o')
plt.xlabel("Time Step")
plt.ylabel("Potential Energy (eV)")
plt.title("Energy Evolution in MD Simulation")
plt.show()
