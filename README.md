# AeroSense AI - Predictive Maintenance for Aircraft Engines

A machine learning project focused on predictive maintenance using NASA's C-MAPSS turbofan engine degradation dataset.

## Project Overview

AeroSense AI explores how aircraft engine sensor measurements can be processed to estimate Remaining Useful Life (RUL) and support predictive maintenance.

The project uses a modular Python structure for data preprocessing, sliding-window generation, normalization, model development, evaluation, and explainability.

## Current Progress

Implemented:
- Loading and initial inspection of the NASA C-MAPSS FD001 dataset
- Engine-level data splitting
- Sliding-window generation for time-series data
- Sensor normalization designed to prevent data leakage
- Unit tests for normalization and windowing

Model training and evaluation are planned next.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Pytest

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.

The FD001 subset contains simulated turbofan engine run-to-failure data.

Dataset files are kept locally and excluded from GitHub.

## Setup

Clone the repository:

```bash
git clone https://github.com/RachanaC03/AeroSense-AI.git
cd AeroSense-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

## Roadmap

- Complete the preprocessing pipeline
- Train a baseline RUL prediction model
- Evaluate model performance
- Add model explainability
- Develop an API and dashboard

## Project Status

Work in progress. Data preparation, sliding-window generation, and leakage-aware normalization have been implemented. Model training and evaluation are not yet complete.