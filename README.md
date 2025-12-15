# ML From Scratch 🧠🛠️

This repository is a collection of **machine learning explorations implemented as standalone scripts**, each focused on understanding core ideas by building and experimenting directly with the mechanics of the model.

The emphasis is on *learning and intuition*, not polished benchmarks. Each file can be read independently and run end-to-end.

---

## Included Explorations

### `lin_reg.py` — Linear & Ridge Regression

* Ordinary Least Squares on synthetic data
* Closed-form ridge regression classifier on MNIST
* Highlights regularization and linear decision boundaries

---

### `convolution_edge.py` — Convolution from Scratch

* Manual 2D convolution on RGB images using NumPy
* Edge-detection kernels applied pixel-by-pixel
* Visualizes how convolution extracts local features

---

### `mnist_dnn.py` — Fully Connected Neural Network (MNIST)

* Multilayer perceptron built with PyTorch
* Trained on MNIST digits
* Demonstrates representation learning beyond linear models

---

### `conv_net.py` — CNNs on CIFAR-10

* Data loading, normalization, and visualization
* Comparison between a fully connected network and a CNN
* Explores how spatial structure improves image classification

---

### `rl_net.py` — Reinforcement Learning (DQN)

* Deep Q-Network trained on CartPole
* Experience replay, target networks, and epsilon-greedy exploration
* Visualizes learned agent behavior

---

### `lm_shakespeare.py` — Language Modeling

* Fine-tunes a GPT-2 model on Tiny Shakespeare
* Tokenization, batching, and causal language modeling
* Generates sample text after training

---

## Tech Stack

* Python
* NumPy
* Matplotlib
* PyTorch / torchvision
* Hugging Face Transformers
* Gymnasium

---

## How to Use

Each file is self-contained. Run them individually after installing the required dependencies for that script.

```bash
python lin_reg.py
python conv_edge.py
python dnn.py
```

---

## Philosophy

This repository is meant to function like a **personal ML lab notebook**:

* Prefer clarity over abstraction
* Favor experimentation over optimization
* Use theory-driven implementations to build intuition

It serves as a foundation for more advanced work in deep learning, reinforcement learning, and research-oriented projects.

