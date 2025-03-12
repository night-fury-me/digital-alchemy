import os
import torch
import numpy as np
from ase import Atoms
from schnetpack.datasets import QM7X
import schnetpack.transform as trn

DATASET_PATH = "data/QM7X_Dataset/QM7X.db"
SAVE_DIR = "./ckpts"
MLFLOW_EXPERIMENT_NAME = "QuantumML-MolDynamics_QM7X"

AVAILABLE_PROPERTIES = ["energy", "forces"]
CUTOFF_RADIUS   = 5.0
N_ATOM_BASIS    = 128
N_INTERACTIONS  = 6
BATCS_SIZE      = 512
LEARNING_RATE   = 1e-4

qm7x_data = QM7X(
    DATASET_PATH,
    batch_size=BATCS_SIZE,
    num_train=8000,
    num_val=1000,
    num_test=1000,
    transforms=[
        trn.ASENeighborList(cutoff=CUTOFF_RADIUS),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=4,
    pin_memory=True,
)

qm7x_data.prepare_data()
qm7x_data.setup()

best_model = torch.load(os.path.join('ckpts/best_model_schnet.ckpt'), map_location="gpu")

