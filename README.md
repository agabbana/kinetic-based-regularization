# A kinetic-based regularization method for data science applications

In this repository we provide an example of code implementation for the interpolator method supplemented with a physics-inspired regularization approach described in [arxiv.2503.04857](https://arxiv.org/abs/2503.04857).


## Installation

You can install the package using the following command in your terminal or Python environment:  
`pip install git+https://github.com/agabbana/kinetic-based-regularization.git`

Alternatively:

1. Clone this repository
2. Open a terminal and navigate inside the repository root folder
3. pip install -e .

## Requirements

- python >= 3.7
- numba  >= 0.57

To execute the demo jupyter notebook you will additionally need `numpy`, `matplotlib`, `jupyter` `scikit-learn`

## Examples

Working examples are provided in the notebook in this repo in the directory `demo-notebooks/`.

## Documentation

#### Constructor

```python
KBRInterpolator(max_iter=5, alpha=0.5, theta_min=1e-7, theta_steps=15, verbose=True)
```

**Parameters:**
- `max_iter` (int, default=5): Maximum iterations for Lagrange multiplier optimization. Higher values may improve accuracy but increase computation time.
- `alpha` (float, default=0.5): Relaxation parameter for theta optimization (0 < alpha ≤ 1). Lower values provide more stable but slower convergence.
- `theta_min` (float, default=1e-7): Minimum threshold for theta values. Prevents numerical instabilities from extremely small theta values.
- `theta_steps` (int, default=15): Number of theta candidates to evaluate during automatic optimization. More steps may find better theta but increase fitting time.
- `verbose` (bool, default=True): Whether to print optimization progress and results.


#### Methods

##### fit(train_data, test_data, theta=None, apply_correction=True)

Fits the KBR interpolator to the training data and optimizes parameters using test data.

**Parameters:**
- `train_data` (numpy.ndarray): Training dataset with shape (n_samples, n_features + 1). Last column should contain target values.
- `test_data` (numpy.ndarray): Test dataset with same structure as train_data, used for parameter optimization.
- `theta` (float, optional): If provided, uses this theta value instead of automatic optimization.
- `apply_correction` (bool, default=True): Whether to apply Lagrange multiplier bias correction.

**Example:**
```python
# Automatic theta optimization with correction
interpolator.fit(train_data, test_data)

# Manual theta setting without correction
interpolator.fit(train_data, test_data, theta=0.05, apply_correction=False)
```

##### predict(x)

Makes predictions on new data using the fitted model.

**Parameters:**
- `x` (numpy.ndarray): Input data with shape (n_samples, dim). Dim-dimensional grid used to make new predictions

**Returns:**
- `numpy.ndarray`: Predicted target values with shape (n_samples,).

**Example:**
```python
predictions = interpolator.predict(validation_data)
print(f"Predictions: {predictions}")
```

## Reference
```

@article{kinetic-based-regularization,
  title   = {A kinetic-based regularization method for data science applications},
  author  = {Ganguly, Abhisek and Gabbana, Alessandro and Rao, Vybhav and Succi, Sauro and Ansumali, Santosh},
  journal = {Machine Learning: Science and Technology},
  publisher = {IOP Publishing},
  volume = {6},
  number = {3},
  pages = {035035},
  month = {aug},
  year = {2025},
  doi = {10.1088/2632-2153/adf93a}
}

```
