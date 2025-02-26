# Project Structure

```
brazilian_sign_language/
├── data/                      # Tracked by DVC
│   ├── raw/                  # Original data
│   │   └── libras+movement  # Movement dataset
│   ├── interim/             # Intermediate data
│   ├── processed/           # Final datasets
│   ├── external/            # Third party sources
│   ├── papers/             # Related research papers
│   └── README.md           # Data documentation
│
├── notebooks/               # Jupyter notebooks
│
├── mlflow/                 # MLflow tracking
│   ├── mlruns/            # Experiment runs
│   └── models/            # Registered models
│
├── models/                 # Model files (tracked by DVC)
│   ├── checkpoints/       # Training checkpoints
│   └── final/            # Production models
│
├── tests/                 # Unit tests
│   ├── data/             # Data processing tests
│   └── models/           # Model tests
│
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock               # Locked dependencies
├── README.md             # Project documentation
└── STRUCTURE.md          # This file
```

## Key Components

### Data Version Control (DVC)
- `.dvc/`: DVC configuration and cache
- `dvc.yaml`: Defines data processing and training pipelines
- `dvc.lock`: Locks pipeline state
- Data and model files are tracked by DVC

### MLflow Integration
- `mlflow/`: Contains experiment tracking data
- Experiments are organized by model type
- Models can be registered and versioned

### Python Environment (uv)
- `pyproject.toml`: Project metadata and dependencies
- `uv.lock`: Locked dependencies for reproducibility
- `.venv/`: Virtual environment (automatically managed by uv)

## Directory Details

### `data/` (DVC-tracked)
- `raw/`: Original, immutable data dumps
- `interim/`: Intermediate data that has been transformed
- `processed/`: Final, canonical data sets for modeling
- `external/`: Data from third party sources
- `papers/`: Related research papers
- `README.md`: Data documentation

### `notebooks/`
Jupyter notebooks for exploration, analysis, and model development:
- `exploration/`: Initial data exploration and analysis
- `preprocessing/`: Data cleaning and preparation steps
- `modeling/`: Model development and evaluation

### `src/`
Source code for use in this project:
- `data/`: Scripts for data operations
- `models/`: Model implementations
- `features/`: Feature extraction code
- `webapp/`: Web application implementation

### `tests/`
Test files for different components of the project

### `configs/`
Configuration files for different components:
- Model hyperparameters
- Preprocessing parameters
- Environment configurations

### `models/`
Saved model files and checkpoints

### `docs/`
Project documentation:
- Data dictionaries
- Model architecture documentation
- API documentation
- Setup guides

### `requirements/`
Dependency management files for different environments

## Best Practices

1. Data Version Control:
   - Use DVC for tracking data and model files
   - Define clear pipelines in `dvc.yaml`
   - Keep data immutable in `raw/`

2. Experiment Tracking:
   - Log all experiments with MLflow
   - Track parameters, metrics, and artifacts
   - Register production models

3. Environment Management:
   - Use uv for dependency management
   - Keep `pyproject.toml` updated
   - Never edit `uv.lock` manually

4. Code Organization:
   - Keep notebooks for exploration
   - Use `src/` for production code
   - Write tests for critical components

5. Documentation:
   - Document data transformations
   - Track experiment configurations
   - Keep API documentation current

## Initial Setup Steps

1. Initialize uv project:
   ```bash
   uv init
   ```

2. Initialize DVC:
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/path  # or other remote
   ```

3. Configure MLflow:
   ```bash
   # Set up MLflow tracking URI in configs/mlflow_configs/
   ```

4. Create virtual environment:
   ```bash
   uv venv
   ``` 