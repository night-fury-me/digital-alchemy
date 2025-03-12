import os
import h5py
import numpy as np
from sys import stdout
from ase import Atoms
from schnetpack.data import ASEAtomsData
from schnetpack.properties import (  # Updated imports
    Z, R, n_atoms, 
    cell, pbc, idx_i, idx_j, Rij
)


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR,  "QM7X_Dataset")
DB_PATH     = os.path.join(DATASET_DIR, "QM7X-2.db")   

# Atom reference energies (used for atomization energy calculations)
EPBE0_atom = {
    6: -1027.592489146,     # Carbon
    17: -12516.444619523,   # Chlorine
    1: -13.641404161,       # Hydrogen
    7: -1484.274819088,     # Nitrogen
    8: -2039.734879322,     # Oxygen
    16: -10828.707468187,   # Sulfur
}

# Define property units
PROPERTY_UNITS = {
    "energy": "eV",
    "atomization_energy": "eV",
    "forces": "eV/Å",
    "dipole": "eÅ",
    "polarizability": "bohr³"
}

# Create ASEAtomsData database
dataset = ASEAtomsData.create(
    DB_PATH,
    distance_unit="Ang",
    property_unit_dict=PROPERTY_UNITS
)

# Buffers for batch processing
atoms_buffer = []
property_buffer = []

stdout.write("\nParsing QM7-x dataset...\n")

# Get list of all HDF5 files
hdf5_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".hdf5")])

# Exclude duplicate molecules (Optional)
def remove_duplicates(molecule_ids):
    try:
        with open("DupMols.dat", "r") as f:
            duplicate_ids = set(line.strip() for line in f)
        return [mol_id for mol_id in molecule_ids if mol_id not in duplicate_ids]
    except FileNotFoundError:
        print("DupMols.dat not found. Skipping duplicate removal.")
        return molecule_ids

# Process each HDF5 file
for filename in hdf5_files:
    file_path = os.path.join(DATASET_DIR, filename)
    
    with h5py.File(file_path, "r") as fMOL:
        molecule_ids = list(fMOL.keys())

        molecule_ids = remove_duplicates(molecule_ids)

        for molid in molecule_ids:
            stdout.write(f"🔹 Processing Molecule {molid}...\n")

            for confid in fMOL[molid].keys():
                try:
                    # Extract atomic properties
                    __Z = np.array(fMOL[molid][confid]["atNUM"])  # Atomic numbers
                    __R = np.array(fMOL[molid][confid]["atXYZ"])  # Atomic positions

                    # Extract total energy (PBE0+MBD energy)
                    energy = float(fMOL[molid][confid]["ePBE0+MBD"][()])
                    
                    # Compute atomization energy (E_atom = E_total - sum(E_atom_reference))
                    atomic_energies = np.array([EPBE0_atom[z] for z in __Z])
                    atomization_energy = energy - np.sum(atomic_energies)

                    forces = np.array(fMOL[molid][confid]["totFOR"])

                    dipole = float(fMOL[molid][confid]["DIP"][()])
                    polarizability = float(fMOL[molid][confid]["mPOL"][()])

                    if forces.shape != (len(__Z), 3):
                        stdout.write(f"Skipping {molid}-{confid}: Invalid force shape {forces.shape}\n")
                        continue

                    atoms = Atoms(numbers=__Z, positions=__R)

                    properties = {
                        Z: __Z,
                        R: __R,
                        "energy": np.array([energy]),                           # Total Energy as (1,) array
                        "atomization_energy": np.array([atomization_energy]),   # Atomization Energy as (1,) array
                        "forces": forces,                                       # Forces as (N,3) array
                        "dipole": np.array([dipole]),                           # Dipole moment as (1,) array
                        "polarizability": np.array([polarizability])            # Polarizability as (1,) array
                    }
                    atoms_buffer.append(atoms)
                    property_buffer.append(properties)

                except KeyError:
                    stdout.write(f"Skipping {molid}-{confid}: Missing required data.\n")
                    continue

stdout.write("\nFinalizing dataset conversion...\n")
dataset.add_systems(property_buffer, atoms_buffer)

stdout.write(f"\nDataset successfully created: {DB_PATH}\n")
