# cheminfo_project

Short overview
---------------
This repository contains code and notebooks for our cheminformatics project focused on HIV activity classification and PDBBind affinity regression. The project uses RDKit for chemistry, scikit-learn / XGBoost / PyTorch Geometric for modelling, and a set of analysis & plotting utilities.

Repository structure
--------------------
- `data/` - CSV datasets used by the notebooks (e.g. `HIV.csv`, `HIV_descriptors.csv`, `LP_PDBBind.csv`).
- `scripts/` - Python helper scripts used by the notebooks:
  - `preprocessing.py` - functions to preprocess PDBBind-like tabular data and to compute RDKit descriptors.
  - `train.py` - model training and evaluation helpers (classification helpers, permutation importance, etc.).
  - `plot.py` - plotting helpers for model results and importances.
  - `atom_permutation_interpretability.py` - functions to compute atom-level perturbations, extract and group substructures, and visualize similarity maps.
- `Plots/` - generated figures .
- `*.ipynb` - analysis and modelling notebooks:
  - `hiv_classification.ipynb` - main notebook for descriptor-based classification (feature selection, PCA/UMAP, model training with LogisticRegression / RandomForest / SVC).
  - `hiv_classification_interpretability.ipynb` - interpretability analyses using fingerprint perturbation and similarity maps; extracts important substructures.
  - `RF_FeatureA.ipynb` - Random Forest regression workflow for PDBBind data (Kd preprocessing, fingerprint extraction, hyperparameter grid-search).
  - `GNN_FeatureA.ipynb` - Graph Neural Network regression.


Libraries used (collected from scripts & notebooks)
--------------------------------------------------
Below is the consolidated list of libraries and packages imported across scripts and notebooks in this repository. This includes both standard library modules and third-party packages.

Standard library / built-ins
- math
- re
- io
- os
- traceback
- collections (defaultdict)
- typing (Optional, Sequence, Union, List)

Third-party packages
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn (sklearn)
- xgboost
- joblib
- rdkit
- networkx
- PIL (Pillow)
- tqdm
- umap-learn (imported as `umap` in a notebook)
- torch (PyTorch)
- torch_geometric (PyG / torch-geometric)
- ogb (ogb.utils.smiles2graph)

Notes and suggestions
---------------------
- RDKit is required for many parts of this project (descriptor calculation, fingerprints, drawing). Install via conda: `conda install -c conda-forge rdkit`.
- PyTorch + PyTorch Geometric + OGB have version-sensitive installation commands; follow their official install guides for compatible CUDA and PyTorch versions.
- `umap-learn` may be required for dimensionality reduction in `hiv_classification.ipynb`.
- `env.yml` shows all library versions needed to run the code.

Authors
-------
- Julian Ostermaier — https://github.com/julianostermaier
- Stephane Dotsenko — https://github.com/Stefoufoune


