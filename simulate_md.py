import os
import torch
import schnetpack as spk
import schnetpack.transform as trn

from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io import Trajectory
from schnetpack.interfaces import SpkCalculator


SIM_DIR = 'simulation'
os.makedirs(SIM_DIR, exist_ok=True)

# Step 1: Load the trained model
model_path = "ckpts/best_model"
model = torch.load(model_path)
model.eval()

# Step 2: Test molecule (CH4)
atoms = Atoms(numbers=[6, 1, 1, 1, 1], positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, -1, -1]])

# Step 3: Attach SchNet Model as ASE Calculator
calculator = SpkCalculator(
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key="energy",
    force_key="forces",
    energy_unit="eV",
    position_unit="Ang",
)

atoms.set_calculator(calculator)

# Step 4: Set Up Molecular Dynamics
timestep    = 1.0      # 1 fs timestep
n_steps     = 100       # Simulate for 100 steps
trajectory_file = f"{SIM_DIR}/trajectory.traj"

# Step 5: Save Trajectory
trajectory = Trajectory(trajectory_file, 'w', atoms)

def log_result():
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Step {dyn.nsteps}: Energy = {energy:.6f} eV")

    atoms.calc.results["energy"] = energy  
    atoms.calc.results["forces"] = forces

    trajectory.write(atoms)


# Step 6: Choose MD Algorithm (Verlet or Langevin)
# dyn = VelocityVerlet(atoms, timestep=timestep)
dyn = Langevin(atoms, timestep=timestep, temperature_K=300, friction=0.02)  # For thermal MD

dyn.attach(log_result, interval=1)

# Step 7: Run the MD Simulation
print("\nRunning Molecular Dynamics Simulation...\n")
dyn.run(n_steps)

print(f"\nSimulation complete! Trajectory saved as {trajectory_file}")