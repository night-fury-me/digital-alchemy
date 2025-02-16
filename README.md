## Project Title: _QuantumML-MolDynamics_ Predicting Molecular Energies and Forces for Advanced Molecular Dynamics Simulations

---

### Prerequisites

-   **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if not already installed.

---

### Setup Instructions

### 1. Create a Conda Environment

A `environment.yml` file is provided to set up the required dependencies. Run the following command to create the Conda environment:

```bash
conda env create -f environment.yml
```

This will create a Conda environment named `alchemy-env` (or the name specified in the environment.yml file).

### 2. Activate the Conda Environment

Activate the environment using:

```bash
conda activate alchemy-env
```

### 3. Run the Dataset Preparation Script

Once the environment is activated, run the prepare-dataset.sh script to automate the dataset preparation process:

```bash
./prepare-dataset.sh
```

This script will:

-   Run download.py to download and convert the dataset to an HDF5 file.
-   Run create-db.py to create a database (.db) file from the downloaded dataset.

---

### Script Details

-   `download.py`: Downloads the dataset and converts it to an HDF5 file.

-   `create-db.py`: Creates a database (.db) file from the HDF5 file.

-   `prepare-dataset.sh`: Automates the execution of `download.py` and `create-db.py`.

---

### Directory Structure

After running the scripts, the project directory will look like this:

```bash
alchemy-proj/
├── data
│   ├── create-db.py
│   ├── download.py
│   ├── prepare-dataset.sh
│   ├── QM7X_Dataset
│   │   ├── 1000.hdf5
│   │   ├── 2000.hdf5
│   │   ├── 3000.hdf5
│   │   ├── 4000.hdf5
│   │   ├── 5000.hdf5
│   │   ├── 6000.hdf5
│   │   ├── 7000.hdf5
│   │   └── 8000.hdf5
│   └── QM7X.db
├── environment.yml
└── README.md
```

---

### Troubleshooting

-   **Conda Environment Issues**: If the environment creation fails, ensure the environment.yml file is correct and try updating Conda:

```bash
conda update conda
```

-   **Permission Denied**: If setup.sh fails to execute, make it executable:

```bash
chmod +x setup.sh
```

-   **Missing Dependencies**: If any Python script fails, ensure all dependencies are installed by checking the environment.yml file.
