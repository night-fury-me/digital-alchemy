import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader

from dig.threedgraph.method import SphereNet
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from QM7x_dataset import QM7xDataset


# List of HDF5 files
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR,  "data/QM7X_Dataset")
hdf5_files  = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".hdf5")])

# Create dataset
dataset = QM7xDataset(hdf5_files)

train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
train_idx, val_idx  = train_test_split(train_idx, test_size=0.1, random_state=42)

train_dataset   = dataset[train_idx]
val_dataset     = dataset[val_idx]
test_dataset    = dataset[test_idx]

epochs          = 200
batch_size      = 64
train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader      = DataLoader(val_dataset, batch_size=batch_size)
test_loader     = DataLoader(test_dataset, batch_size=batch_size)


# 3. Training Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SphereNet(
    energy_and_force=True,
    cutoff=5.0,
    num_layers=4,
    hidden_channels=128,
    out_channels=1,
    use_node_features=False
).to(device)

optimizer = torch.optim.AdamW(  
    model.parameters(),
    lr = 0.0005,
    weight_decay = 1e-5,
    eps = 1e-6
)

# StepLR scheduler: Reduce LR by 0.5 every 15 epochs
lr_decay_factor = 0.5
lr_decay_step_size = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

energy_criterion = torch.nn.L1Loss()
force_criterion  = torch.nn.L1Loss()

# 5. Evaluation Function
def evaluate(loader, desc="Evaluating"):
    model.eval()
    energy_mae, force_mae = 0, 0
    original_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    
    # Disable parameter gradients
    for p in model.parameters():
        p.requires_grad = False

    try:
        with torch.no_grad():
            batch_bar = tqdm(loader, desc=desc, leave=False)
            for data in batch_bar:
                data = data.to(device)
                data.pos.requires_grad = True
                
                # Forward pass
                energy_pred = model(data)
                
                # Force calculation
                force_pred = -torch.autograd.grad(
                    energy_pred.sum(),
                    data.pos,
                    create_graph=False
                )[0]
                
                # Update metrics
                current_e_mae = F.smooth_l1_loss(energy_pred, data.y, beta=1.0).item()
                current_f_mae = F.smooth_l1_loss(force_pred, data.forces, beta=1.0).item()
                
                energy_mae += current_e_mae * data.num_graphs
                force_mae += current_f_mae * data.num_graphs
                
                # Update progress bar
                batch_bar.set_postfix({
                    'energy_mae': f"{current_e_mae:.4f}",
                    'force_mae': f"{current_f_mae:.4f}"
                })
    finally:
        # Restore original gradient settings
        for n, p in model.named_parameters():
            p.requires_grad = original_requires_grad[n]
    
    return energy_mae / len(loader.dataset), force_mae / len(loader.dataset)

# 6. Training Loop
best_val_mae = float('inf')
epoch_bar = tqdm(range(1, epochs+1), desc="Total training progress", position=0)

for epoch in epoch_bar:
    # Training with batch progress
    model.train()
    total_loss = 0
    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
    
    for data in batch_bar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        energy_pred = model(data)
        force_pred = -torch.autograd.grad(
            energy_pred.sum(),
            data.pos,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Calculate losses
        loss_energy = F.smooth_l1_loss(energy_pred, data.y, beta=1.0)
        loss_force  = F.smooth_l1_loss(force_pred, data.forces, beta=1.0)
        loss = loss_energy + loss_force
        
        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        
        # Update progress
        total_loss += loss.item() * data.num_graphs
        batch_bar.set_postfix({
            'batch_loss': loss.item(),
            'energy_loss': loss_energy.item(),
            'force_loss': loss_force.item()
        })
    
    # Validation
    val_energy, val_force = evaluate(val_loader)
    epoch_bar.set_postfix({
        'train_loss': total_loss / len(train_loader.dataset),
        'val_energy': f"{val_energy:.4f}",
        'val_force': f"{val_force:.4f}"
    })
    
    # Save best model
    if val_energy < best_val_mae:
        best_val_mae = val_energy
        torch.save(model.state_dict(), f'ckpts/spherenet_best_model_at_epoch_{epoch}.pth')
    
    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), f'ckpts/spherenet_model_at_epoch_{epoch}.pth')

