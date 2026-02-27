# MLE Optimizer — Numerical Methods from Scratch

> Estimating parameters of a Normal distribution using Maximum Likelihood Estimation (MLE),
> implemented with three classical numerical optimization methods compared for speed and accuracy.

---

## 📌 Project Overview

This project demonstrates how to find the **Maximum Likelihood Estimates** of μ (mean) and σ (standard deviation)
for a Normal distribution using three numerical methods built entirely from scratch:

| Method | Use Case | Iterations (this project) |
|---|---|---|
| Newton's Method | Fast, derivative-based optimization for μ | **2** |
| Bisection Method | Safe, derivative-free root finding for μ | **24** |
| Golden Section Search | Bounded search for σ (no derivatives needed) | **35** |

All three methods converge to the **same answer** demonstrating the classic tradeoff between
convergence speed and information requirements.

---

## 📊 Results

```
Parameter    True Value    MLE Estimate    Method
─────────────────────────────────────────────────
μ (mean)     5.0           4.792307        Newton's Method
σ (std)      2.0           1.807232        Golden Section
```

---

## 🖼️ Convergence Comparison

<img width="1390" height="495" alt="image" src="https://github.com/user-attachments/assets/28e2d7e1-0133-4c90-a5c7-c9dfc823ef0b" />


- **Newton (red)** — converges in 2 iterations using second-order derivative information
- **Bisection (blue)** — zigzags toward the answer, guaranteed convergence in 24 iterations
- **Golden Section (orange)** — derivative-free, converges in 35 iterations using golden ratio bracketing

---

## 🧠 Key Concepts Covered

- Maximum Likelihood Estimation (MLE) for Normal distribution
- Log-likelihood function derivation
- Newton's method using first and second derivatives
- Bisection method on the derivative (root finding)
- Golden Section search for bounded parameter optimization
- Convergence analysis and method comparison

---

## 📁 Project Structure

```
mle-optimizer/
│
├── notebook/
│   └── MLE.ipynb    ← full walkthrough with explanations
│
├── src/
│   └── methods.py             ← clean reusable implementations
│
├── images/
│   └── convergence_plot.png   ← convergence comparison plot
│
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/techie-code/mle-optimizer.git
cd mle-optimizer
pip install numpy matplotlib scipy
jupyter notebook notebook/mle_optimizer.ipynb
```

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 💡 Why This Project?

In real-world data science and ML, optimization is everywhere , training neural networks,
fitting statistical models, tuning hyperparameters. Understanding *how* optimizers work
under the hood (not just calling `scipy.optimize`) is what separates strong data scientists
from those who just use black boxes.

This project was built as part of my Masters in Statistical Data Science to bridge
numerical methods theory with practical statistical computing.

---

## 👤 Author

**Aishwarya Lakshmi S**
Masters in Statistical Data Science | Ex-IBM Data Engineer (3 years)

- LinkedIn: [[linkedin](https://uk.linkedin.com/in/aishwarya-lakshmi-s-ba9ab4179)]
- GitHub: [[github](https://github.com/techie-code)]
