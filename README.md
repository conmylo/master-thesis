# Continuous Implicit Authentication System

This repository contains the implementation of a Continuous Implicit Authentication System. The system leverages natural language processing (NLP), machine learning techniques, and one-class support vector machines (OC-SVM) for anomaly detection and authentication in interactive environments.

## Table of Contents
1. [Features](#features)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Training Models](#training-models)
5. [Testing Models](#testing-models)
6. [Running the Streamlit Application](#running-the-streamlit-application)
7. [Configuration](#configuration)
8. [Data Requirements](#data-requirements)
9. [Acknowledgments](#acknowledgments)

## Features
- **Training**: Train OC-SVM models on text data to create user-specific authentication models.
- **Testing**: Evaluate trained models using cross-validation techniques such as Leave One Subject Out (LOSO).
- **Real-Time Authentication**: Use a Streamlit-based web app to perform real-time authentication with confidence scoring and decision-making.

## Repository Structure
```plaintext
root/
├── app.py                    # Streamlit application
├── src/                      # Source code for core functionalities
│   ├── config.py             # Configuration variables
│   ├── extract_features.py   # Feature extraction logic
│   ├── test_model.py         # Testing logic
│   ├── train_model.py        # Training logic
│   ├── utils.py              # Utility functions
├── models/                   # Directory for trained models
├── data/                     # Directory for dataset storage
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation
```

## Installation

### Prerequisites
Ensure you have Python 3.8 or later installed.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/continuous-authentication.git
   cd continuous-authentication
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Models
To train models for each user:

1. Ensure the `data/` directory contains CSV files with the following columns:
   - `author`: The username of the text author.
   - `content`: The text data for training.

2. Run the training script:
   ```bash
   python src/train_model.py
   ```

3. Models will be saved in the `models/` directory, named as:
   ```plaintext
   user_{username}_nu_{nu}_gamma_{gamma}_model.pkl
   user_{username}_nu_{nu}_gamma_{gamma}_scaler.pkl
   user_{username}_nu_{nu}_gamma_{gamma}_distance.pkl
   ```

## Testing Models
To test the trained models:

1. Ensure the `data/` directory contains test CSV files named `user_{username}_test.csv` for each user.
2. Run the testing script:
   ```bash
   python src/test_model.py
   ```
3. The script calculates evaluation metrics such as False Acceptance Rate (FAR), False Rejection Rate (FRR), and more.

## Running the Streamlit Application
The Streamlit app provides a web interface for real-time authentication.

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Access the app in your browser at `http://localhost:8501/`.

3. Use the dropdown menu to select a username and input a text prompt for authentication. The app will display whether access is granted or denied along with the certainty score.

## Configuration
All configurable parameters, such as the model directory, hyperparameters (`nus` and `gammas`), and confidence thresholds, are defined in `src/config.py`. Update these values as needed.

Example:
```python
MODEL_DIR = "models/"
NUS = [0.001, 0.005, 0.01]
GAMMAS = [0.05, 0.1, 0.2]
CONFIDENCE_THRESHOLD = 0.3
```

## Data Requirements
- **Training Data**: Ensure your dataset contains sufficient samples per user for robust model training.
- **Testing Data**: Each user should have a separate test dataset for evaluation.

Data must be in CSV format with at least the following columns:
- `author`: Username of the text author.
- `content`: Text content for training or testing.

## Acknowledgments
This project utilizes concepts and techniques in natural language processing, one-class SVM, and real-time authentication. We acknowledge the open-source libraries and research that made this work possible.

For any questions or feedback, please create an issue or contact the repository maintainer at [your-email@example.com].
