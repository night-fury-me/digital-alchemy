> _This repository contains Digital Alchemy Project (WiSe24/25 at FAU) and uses [QM7-X dataset](https://zenodo.org/records/4288677)._

## Project Title: **"_QuantumML-MolDynamics_"** Predicting Molecular Energies and Forces for Advanced Molecular Dynamics Simulations

<img src="images/banner.png" alt="Project Banner" width="100%">

---

### **Overview**

This project trains a machine learning model using the **QM7-X dataset** to predict **molecular energies and forces**. The trained model is then used in **Molecular Dynamics (MD) simulations** to study atomic motion under various conditions.

---

### **Prerequisites**

Ensure that you have the following installed on your system:

-   [Docker](https://docs.docker.com/get-docker/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

To verify that Docker is installed, run:

```bash
docker --version
```

If you want to use GPU support, ensure the NVIDIA runtime is set up by running:

```bash
docker run --rm --gpus all nvidia/cuda:11.7.1-base nvidia-smi
```

This should display information about your GPU.

---

### **Building the Docker Image**

Clone your repository and navigate to the project directory:

```bash
git clone https://github.com/night-fury-me/digital-alchemy.git
cd digital-alchemy
```

Then, build the Docker image using:

```bash
docker build -t digital_alchemy_img .
```

This will create an image named `digital_alchemy_img`.

---

### **Running the MLflow Server Inside a Docker Container**

To start the MLflow server inside a Docker container and expose the MLflow UI on port 5000, run:

```bash
docker run -d --gpus all -p 5000:5000 -v $(pwd)/mlruns:/workspace/mlruns -w /workspace --name mlflow_server digital_alchemy_img mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /workspace/mlruns
```

Once the server is running, you can access the MLflow UI at:

```
http://localhost:5000
```

---

### **Dataset Preparation**

Run the dataset preparation script:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace digital_alchemy_img bash data/prepare-dataset.sh
```

This script will:

-   Run `download.py` to download and convert the dataset to an HDF5 file.
-   Run `create-db.py` to create a database (.db) file from the dataset.

---

### **Directory Structure**

After running the scripts, the project directory will look like this:

```bash
digital-alchemy/
├── ckpts
│   └── best_model
├── data
│   ├── create-db.py
│   ├── download.py
│   ├── prepare-dataset.sh
│   └── QM7X_Dataset
│       ├── 1000.hdf5
│       ├── 2000.hdf5
│       ├── 3000.hdf5
│       ├── 4000.hdf5
│       ├── 5000.hdf5
│       ├── 6000.hdf5
│       ├── 7000.hdf5
│       ├── 8000.hdf5
│       └── QM7X.db
├── energy-vs-time.py
├── environment.yml
├── evaluate.py
├── lightning_logs
│   └── version_1
│       ├── checkpoints
│       │   └── epoch=83-step=5292.ckpt
│       ├── events.out.tfevents.1739748819.redStation.1030311.0
│       └── hparams.yaml
├── mlruns
│   ├── 0
│   │   └── meta.yaml
│   ├── 595388344762858645
│   └── models
├── paper-presentation
│   ├── BiM-Network.ipynb
│   ├── dataset-explore.ipynb
│   └── README.md
├── qm7x_ase_calculations
│   └── test_molecule.xyz
├── README.md
├── requirements.txt
├── simulate_md.py
├── simulation
│   └── trajectory.traj
├── split.npz
├── splitting.lock
└── train.py
```

### **Running the Training Script (`train.py`) in a Docker Container**

After the MLflow server is running, open another terminal and execute:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" digital_alchemy_img python train.py
```

---

### **Verifying MLflow Logging**

After running the training script, check the MLflow UI (`http://localhost:5000`) to ensure experiment metrics, parameters, and logs are recorded.

---

### **Stopping the MLflow Server**

To stop the MLflow server, find the running container ID using:

```bash
docker ps
```

Then stop the container using:

```bash
docker stop <container-id>
```

---

### **Training Details**

-   Uses `SchNet` as the base neural network.
-   Optimized using `AdamW optimizer`.
-   Loss function balances energy (MSE loss) and force predictions.
-   Logs training results in MLflow.

---

### **Evaluating the Model**

Once training is complete, evaluate the model on unseen test molecules:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" digital_alchemy_img python evaluate.py
```

---

### **Running Molecular Dynamics (MD) Simulations**

To run MD simulations:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace --network="host" digital_alchemy_img python simulate_md.py
```

#### **Debugging MD Issues:**

If energy remains **constant instead of fluctuating**, try:

-   Reducing the timestep (`0.5 fs` instead of `1 fs`).
-   Increasing Langevin friction (`0.1` instead of `0.02`).
-   Printing force values:
    ```python
    print("Forces at Step 100:", atoms.get_forces())
    ```
-   Printing temperature:
    ```python
    print("Temperature at Step 100:", atoms.get_temperature())
    ```

---

### **Visualizing MD Results**

#### **Install ASE (Atomic Simulation Environment)**

```bash
pip install ase
```

#### **View Atomic Motion in ASE GUI**

```bash
ase gui trajectory.traj
```

#### **Plot Energy vs. Time**

```bash
python energy-vs-time.py
```

#### **Extract MD Energies from Trajectory**

```python
from ase.io import Trajectory
traj = Trajectory("simulation/trajectory.traj")
for step, atoms in enumerate(traj):
    print(f"Step {step}: Energy = {atoms.get_potential_energy()} eV")
```

---

### **Troubleshooting**

#### **1. Permission Denied for `prepare-dataset.sh`**

```bash
chmod +x data/prepare-dataset.sh
```

---

### **References**

-   Christensen et al., (2020). QM7-X, a comprehensive dataset. _Scientific Data_, 7(1), 1-7. https://doi.org/10.1038/s41597-020-0473-z
-   Schütt et al., (2023). SchNetPack 2.0: _A neural network toolbox for atomistic machine learning._ https://doi.org/10.1063/5.0138367
