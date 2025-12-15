# handwritten_digits

This project is a single-file, math-focused exploration of linear regression and ridge regression. It is designed to build intuition for classical machine learning methods without relying on high-level training frameworks.

---

* **Ordinary Least Squares (OLS)** on synthetic 1D data

  * Generates noisy linear data
  * Solves for slope and intercept using the closed-form solution
  * Visualizes the fitted line

* **Ridge Regression Classifier on MNIST**

  * Loads MNIST using torchvision
  * Flattens images and one-hot encodes labels
  * Trains a multiclass linear classifier with L2 regularization
  * Evaluates accuracy on a held-out test set

---

## Tech Stack

* Python
* NumPy
* Matplotlib
* torchvision (for MNIST loading)
