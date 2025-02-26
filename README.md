# Brazilian Sign Language Recognition Project

This is a collaborative data science project under the Omdena São Paulo Chapter focused on Brazilian Sign Language (Libras) recognition. The project aims to develop machine learning models that can classify sign language videos into corresponding Portuguese words.

## Project Overview

See [STRUCTURE.md](STRUCTURE.md) for detailed project organization.

## Setup Instructions

### Prerequisites

1. Python 3.11 or higher
2. uv package manager (recommended installation via pip: `pip install uv`)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage.git
   cd SaoPauloBrazilChapter_BrazilianSignLanguage
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```
   This installs core dependencies (DVC, MLflow, etc.). For additional dependencies:
   ```bash
   uv sync --extra <group>  # Example: uv sync --extra data
   ```
   See `pyproject.toml` for available dependency groups (data, model, app).

3. Using the environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   ```

   Or run commands directly without activation:
   ```bash
   uv run python your_script.py
   # or for Jupyter:
   uv run jupyter notebook
   ```

4. Adding new dependencies:
   ```bash
   # Add to core dependencies
   uv add numpy

   # Add to a specific group
   uv add --extra data opencv-python  # For data processing tools
   uv add --extra model tensorflow    # For ML frameworks
   uv add --extra app fastapi         # For web app development
   ```

## Project Structure

```
SaoPauloBrazilChapter_BrazilianSignLanguage/
├── data/                  # Data files (tracked by DVC)
│   ├── raw/              # Original data
│   ├── interim/          # Intermediate processing
│   ├── processed/        # Final datasets
│   ├── external/         # Third party data
│   └── papers/           # Related research
├── notebooks/            # Jupyter notebooks
├── mlflow/              # MLflow tracking (will be configured with DagsHub)
├── models/              # Model files (tracked by DVC)
└── tests/              # Unit tests
```

See [STRUCTURE.md](STRUCTURE.md) for complete structure details.

## Data

Data files will be version controlled using DVC. Setup instructions will be added as we progress.

## Experiments

Experiments will be tracked using MLflow through DagsHub. Setup instructions will be added as we progress.

## Contributing

This is a collaborative project under the Omdena São Paulo Chapter. Please coordinate with the chapter leads for contribution guidelines.

## License

[To be determined]