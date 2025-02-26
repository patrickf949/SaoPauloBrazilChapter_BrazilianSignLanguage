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

2. Install project dependencies:
   ```bash
   uv sync
   ```
   This will:
   - Create a virtual environment in `.venv/`
   - Install all dependencies from `pyproject.toml`
   - Create/update `uv.lock` for reproducible installations

3. Using the environment:
   Option 1 - Run commands directly:
   ```bash
   uv run python your_script.py
   # or for Jupyter:
   uv run jupyter notebook
   ```
   
   Option 2 - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   ```

4. Adding new dependencies:
   ```bash
   uv add package_name  # Example: uv add scikit-learn
   ```
   This will:
   - Add the package to `pyproject.toml`
   - Update `uv.lock`
   - Install the package in your environment

## Project Structure

```
SaoPauloBrazilChapter_BrazilianSignLanguage/
├── data/                  # Tracked by DVC
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
├── tests/               # Unit tests
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock             # Locked dependencies for reproducibility
└── ...
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