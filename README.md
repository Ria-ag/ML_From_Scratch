# handwritten_digits

This project is a single-file, math-focused exploration of **linear regression and ridge regression** implemented directly in NumPy. It is designed to build intuition for classical machine learning methods without relying on high-level training frameworks.

---

## What’s Included

* **Ordinary Least Squares (OLS)** on synthetic 1D data

  * Generates noisy linear data
  * Solves for slope and intercept using the closed-form solution
  * Visualizes the fitted line

* **Ridge Regression Classifier on MNIST**

  * Loads MNIST using torchvision
  * Flattens images and one-hot encodes labels
  * Trains a multiclass linear classifier with L2 regularization
  * Evaluates accuracy on a held-out test set

All code lives in a single Python file to keep the focus on the underlying math and data flow.

---

## Tech Stack

* Python
* NumPy
* Matplotlib
* torchvision (for MNIST loading)

---

## Why This Project

This project emphasizes:

* Closed-form solutions over gradient descent
* Interpretability and simplicity
* Understanding how regularization behaves in high-dimensional settings

It serves as a foundation for more complex models later on.
