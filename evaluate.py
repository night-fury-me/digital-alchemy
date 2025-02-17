import os
import torch
import schnetpack as spk
import schnetpack.transform as trn
from ase import Atoms, io
from schnetpack.data import ASEAtomsData
from schnetpack.interfaces import AtomsConverter, SpkCalculator

# Step 1: Set Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load Trained Model
model_path = "ckpts/best_model" 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

print(f"Loading model from: {model_path}")
best_model = torch.load(model_path, map_location=device)
best_model.eval()  # Set to evaluation mode

# Step 3: Set Up AtomsConverter (For Model Inference)
converter = AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    dtype=torch.float32,
    device=device,
)

# Step 4: Load Sample Structure from QM7-X Dataset
dataset_path = "data/QM7X_Dataset/QM7X.db"
qm7x_data = ASEAtomsData(dataset_path)

# Pick a test molecule (adjust index for different molecules)
structure = qm7x_data[100]  
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)

actual_energy = structure["energy"].item()
actual_forces = structure["forces"]

print("\nGround Truth Values (QM7-X Dataset):")
print(f"Actual Energy: {actual_energy:.6f} eV")
print(f"Actual Forces:\n{actual_forces}")

# Step 5: Convert Atoms Object and Perform Prediction
inputs = converter(atoms)
results = best_model(inputs)

print("\nPrediction Results:")
print(f"Energy: {results['energy'].item()} eV")
print(f"Forces:\n{results['forces']} eV/Å")

# Step 6: Integrate Model with ASE Calculator
calculator = SpkCalculator(
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key="energy",
    force_key="forces",
    energy_unit="eV",
    position_unit="Ang",
)

atoms.set_calculator(calculator)

# Step 7: Compute Energy and Forces with ASE
print("\nASE Calculations:")
print(f"Energy from ASE: {atoms.get_potential_energy()} eV")
print(f"Forces from ASE:\n{atoms.get_forces()} eV/Å")

# Step 8: Save Molecule Structure for Visualization
ase_dir = "qm7x_ase_calculations"
os.makedirs(ase_dir, exist_ok=True)
molecule_path = os.path.join(ase_dir, "test_molecule.xyz")
io.write(molecule_path, atoms, format="xyz")

print(f"\nStructure saved to {molecule_path}.")
