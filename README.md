# Dermatological Diagnosis using XAI with Inception V3 and Grey Wolf Optimizer

This project implements a dermatological diagnosis system using deep learning with explainable AI capabilities. It uses Inception V3 for feature extraction, Grey Wolf Optimization for feature selection, and LIME for generating visual explanations of model predictions.

## Features

- Utilizes the ISIC 2019 dataset for skin lesion classification
- Implements Inception V3 for deep feature extraction
- Applies Grey Wolf Optimization for feature selection
- Provides explainable results using LIME (Local Interpretable Model-agnostic Explanations)
- Handles data preprocessing, augmentation, and class imbalance

## Requirements

See `requirements.txt` for a full list of dependencies. Major requirements include:
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- LIME
- Imbalanced-learn
- OpenCV
- Matplotlib

## Installation

1. Clone the repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset

This project uses the ISIC 2019 dataset. You need to download it from:
[ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/)

Update the data path in `main.py` to point to your local copy of the dataset.

## Usage

1. Update the data directory path in `main.py`
2. Run the main script:
   ```
   python main.py
   ```
3. When prompted, choose a model type to run:
   - Option 1: Inception V3 transfer learning model
   - Option 2: Simple neural network model
   - Option 3: Continue from last checkpoint

## Project Structure

- `main.py`: Main driver script
- `data_processor.py`: Handles data loading, preprocessing, and augmentation
- `model_builder.py`: Builds and trains the Inception V3 based model
- `model_fix.py`: Alternative model definition with simpler architecture
- `gwo_optimizer.py`: Implements Grey Wolf Optimizer for feature selection
- `xai_explainer.py`: Implements LIME for generating model explanations
- `debugging.py`: Utilities for evaluating and debugging models
- `tf_utils.py`: Utilities for TensorFlow compatibility and warning suppression

## Output

The system will:
1. Load and preprocess the ISIC 2019 dataset
2. Train an Inception V3 based model
3. Apply Grey Wolf Optimization for feature selection
4. Generate LIME explanations for sample predictions
5. Save the optimized model and visualizations

## Citation

If you use this code in your research, please cite:
```
@article{DermatologicalDiagnosisXAI2023,
  title={Dermatological Diagnosis using XAI with Inception V3 and Grey Wolf Optimizer},
  author={Your Name},
  journal={},
  year={2023}
}
```

## Understanding Grey Wolf Optimization

### What is Grey Wolf Optimizer?

Grey Wolf Optimizer (GWO) is a meta-heuristic algorithm inspired by the leadership hierarchy and hunting mechanism of grey wolves in nature. In this project, we use GWO for feature selection to improve model efficiency and interpretability.

### Interpreting GWO Output

The GWO algorithm is designed to **minimize** an objective function. However, in machine learning, we typically want to **maximize** metrics like accuracy. To handle this, we negate the accuracy in the fitness function:
