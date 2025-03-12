import os
import torch
import torchmetrics
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn


# Define paths
DATASET_PATH = "data/MD17/MD17.db"
# DATASET_PATH = "data/QM7X_Dataset/QM7X.db"
SAVE_DIR = "./ckpts"
MLFLOW_EXPERIMENT_NAME = "QuantumML-MolDynamics_QM7X"

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# Define dataset properties
AVAILABLE_PROPERTIES = ["energy", "forces"]
CUTOFF_RADIUS   = 5.0
N_ATOM_BASIS    = 128
N_INTERACTIONS  = 6
BATCS_SIZE      = 512
LEARNING_RATE   = 1e-4

# # Step 1: Load QM7-X dataset
# dataset = spk.data.AtomsDataModule(
#     DATASET_PATH,
#     batch_size=BATCS_SIZE,
#     num_train=8000,
#     num_val=1000,
#     num_test=1000,
#     # distance_unit='Ang',
#     # property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
#     transforms=[
#         trn.ASENeighborList(cutoff=CUTOFF_RADIUS),
#         trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
#         trn.CastTo32()
#     ],
#     num_workers=4,
#     pin_memory=True,
# )

# dataset.prepare_data()
# dataset.setup()

dataset = spk.datasets.MD17(
    DATASET_PATH,
    molecule="ethanol",
    batch_size=BATCS_SIZE,
    num_train=80000,
    num_val=1000,
    num_test=1000,
    distance_unit='Ang',
    property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
    transforms=[
        trn.ASENeighborList(cutoff=CUTOFF_RADIUS),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=20,
    pin_memory=True,
    split_file="data/MD17/split.npz"
)

dataset.prepare_data()
dataset.setup()

# Step 2: Define PaiNN Model 
pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF_RADIUS)

model = spk.representation.SchNet(
    n_atom_basis=N_ATOM_BASIS,
    n_interactions=N_INTERACTIONS,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
)

# model = spk.representation.FieldSchNet(
#     n_atom_basis = N_ATOM_BASIS,
#     n_interactions = N_INTERACTIONS,
#     radial_basis = radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
# )

# model = spk.representation.PaiNN(
#     n_atom_basis=N_ATOM_BASIS,
#     n_interactions=N_INTERACTIONS,
#     radial_basis=radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
# )

# model = spk.representation.SO3net(
#     n_atom_basis=N_ATOM_BASIS,
#     n_interactions=N_INTERACTIONS,
#     lmax=1,
#     radial_basis=radial_basis,
#     cutoff_fn=spk.nn.CosineCutoff(CUTOFF_RADIUS),
# )

# Step 3: Define Output Modules 
pred_energy = spk.atomistic.Atomwise(n_in=N_ATOM_BASIS, output_key="energy")
pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")

# Step 4: Assemble Neural Network Potential
nnpot = spk.model.NeuralNetworkPotential(
    representation= model,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
    ]
)

# Step 5: Define Loss Functions 
output_energy = spk.task.ModelOutput(
    name="energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError(),
        "RMSE": torchmetrics.MeanSquaredError(squared=False) 
    }
)

output_forces = spk.task.ModelOutput(
    name="forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError(),
        "RMSE": torchmetrics.MeanSquaredError(squared=False) 
    }
)

# Step 6: Define Training Task 
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": LEARNING_RATE},
)

# Define Trainer (
logger = pl.loggers.TensorBoardLogger(save_dir="./tf-logs")

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(SAVE_DIR, "best_model_schnet_md17.ckpt"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    log_every_n_steps=10,
    default_root_dir="./tf-logs",
    max_epochs=200,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# Device setup 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, training on CPU.")

# Step 8: Train the Model 
trainer.fit(task, datamodule=dataset)


print("\nTraining complete! Model logged to tensorboard.")
