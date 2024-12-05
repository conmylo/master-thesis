# Continuous Implicit Authentication System

## Overview

This project implements a Continuous Implicit Authentication System designed to classify user authenticity based on text input. The system is built with functionalities for training, testing, and running a web-based application using Streamlit. The core models are built using One-Class SVMs (OC-SVMs) and employ techniques such as weighted majority voting and confidence level adjustments to ensure robust authentication.

---

## Repository Structure

```
project_root/
├── app.py                  # Streamlit application for user authentication
├── src/                   # Source code directory
│   ├── __init__.py        # Package initializer
│   ├── main.py            # Main entry point for training and testing
│   ├── config.py          # Configuration variables (paths, parameters)
│   ├── train_model.py     # Model training logic
│   ├── test_model.py      # Testing and evaluation logic
│   ├── extract_features.py # Feature extraction from text
│   ├── utils.py           # Utility functions (loading models, scalers, etc.)
├── models/                # Directory to store trained models
├── data/                  # Training and testing datasets
└── README.md              # Documentation
```

---

## Prerequisites

- **Python 3.8+**
- Required packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`
- Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Configuration

The configuration parameters are stored in `src/config.py`. Key parameters include:

- `DATA_PATH`: Path to the training dataset.
- `TEST_DATA_PATH`: Path to the testing dataset.
- `MODEL_DIR`: Directory where models will be stored.
- `NUS`: List of `nu` values for OC-SVM models.
- `GAMMAS`: List of `gamma` values for OC-SVM models.
- `CONFIDENCE_THRESHOLD`: Threshold for confidence level in decision-making.

---

## Usage

### Training

To train models using the provided dataset:

1. Ensure the dataset is placed at the path defined in `DATA_PATH`.
2. Run the following command:

```bash
python src/main.py train
```

This will:
- Train OC-SVM models using the defined `NUS` and `GAMMAS`.
- Save the trained models and scalers in the `MODEL_DIR` directory.

### Testing

To test models and evaluate their performance:

1. Ensure test data is placed at the path defined in `TEST_DATA_PATH`.
2. Run the following command:

```bash
python src/main.py test
```

This will:
- Perform testing using the trained models.
- Display metrics like FAR, FRR, and others.

### Running the Application

To launch the Streamlit web application:

1. Ensure trained models are available in the `MODEL_DIR` directory.
2. Run the following command:

```bash
streamlit run app.py
```

3. The application will be accessible at `http://localhost:8501`.
4. Use the dropdown to select a username and input a text prompt for authentication.

---

## How the System Works

### Authentication Workflow

1. **Input**: The user provides a username and a text prompt.
2. **Model Loading**: The app loads all relevant OC-SVM models and scalers for the user.
3. **Feature Extraction**: The text input is processed into numerical features.
4. **Weighted Voting**:
    - Each model predicts authenticity with a certainty score.
    - Weighted majority voting aggregates the predictions.
5. **Decision**: The system grants or denies access based on the aggregated result.

### Key Techniques

- **Weighted Majority Voting**: Ensures robust decision-making by considering both predictions and their certainty.
- **Confidence Level Adjustment**: Dynamically adjusts based on consecutive correct or incorrect predictions.

---

## Troubleshooting

- **Error: Model for user `USERNAME` not found**:
    - Ensure the model for the specified username exists in the `MODEL_DIR`.
    - Verify that the training process has been completed for the user.

- **App Fails to Start**:
    - Check that all dependencies are installed.
    - Verify the configuration in `src/config.py`.

---

## Future Extensions

- Integrate support for additional feature extraction methods.
- Enhance the confidence level function with advanced analytics.
- Expand to support more languages and longer text prompts.

---

## Contact

For questions or contributions, please contact the repository maintainer.
