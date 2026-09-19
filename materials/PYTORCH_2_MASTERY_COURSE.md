# PyTorch 2 Mastery: Learn by Building

> A first-principles, practice-heavy course from tensors to distributed training.
>
> **Target:** PyTorch 2.14.0 (verified 2026-09-19)  
> **Prerequisite:** Comfortable Python; high-school algebra; willingness to debug.  
> **Rule:** Do not merely read this course. Type every example, predict its output, break it, repair it, and complete the assignments without copying.

## What this course is designed to fix

Watching someone train a neural network produces recognition: “I understand that code.” Building one from an empty file requires recall, judgment, and debugging. Those are different skills.

This course uses a repeated loop:

1. **Why:** identify the problem a feature solves.
2. **Model:** understand the smallest useful mental model.
3. **Mechanism:** see what PyTorch does internally.
4. **Example:** run one small, complete example.
5. **Practice:** solve several exercises from memory.
6. **Build:** combine the ideas in a project.
7. **Retrieve:** redo key tasks later without notes.

You learn PyTorch when you can make correct choices in a new project—not when an API looks familiar.

---

## How to use the course

### The practice protocol

For each chapter:

- Read only through the worked example.
- Close the file and reproduce the example from memory.
- Complete the core assignments before moving on.
- If stuck for 20 minutes, read only the smallest relevant hint or documentation page.
- Keep a `mistakes.md` file. Record the symptom, root cause, and the rule that would have prevented it.
- Revisit the marked retrieval exercises after **1 day, 1 week, and 1 month**.

An assignment is complete only when:

- it runs from a fresh process;
- shapes and dtypes are asserted at important boundaries;
- training and evaluation behavior are separated correctly;
- you can explain every tensor dimension;
- you can rebuild the core without looking at the solution.

Most assignments use three labels:

- **Given:** the exact starting input or situation;
- **Build/Do:** the required work and important restrictions;
- **Pass when:** observable acceptance criteria. If those checks pass and you can explain the result, the assignment is finished.

### Reading and running examples

Start with the installation section at the end, then return here. Chapters 0–12 can be learned on a CPU. CUDA performance experiments and multi-GPU sharding are explicitly hardware-dependent; understanding them does not require buying a GPU.

In code, use these shared imports unless a block includes its own imports:

```python
import torch
from torch import nn
import torch.nn.functional as F
```

A **worked example** defines its inputs. A short **continuation** reuses the objects introduced immediately above it. An **integration fragment** uses an existing model, loader, or optimizer; its surrounding text names those prerequisites. Do not paste unrelated chapter fragments into one giant script: names such as `x` deliberately change meaning between lessons. Shapes like `[B,T,D]` are explanatory notation, not Python lists to feed to a constructor.

At each equation, first identify the input, output, and one concrete number. You do not need prior calculus: the next chapter introduces the only derivative idea needed to begin.

### What is covered

The core path is framework mastery: tensors, autograd, modules, data, training, debugging, performance, compilation, distributed execution, and export. CNNs and transformers are used as vehicles. This is not a complete course in statistics, linear algebra, computer vision, NLP, or MLOps.

The [PyTorch Cheatsheet by Rohit Bandaru](https://rohitbandaru.github.io/notes/pytorch-cheatsheet/) is used as a **coverage checklist**, not as the teaching text. That page intentionally assumes prior PyTorch experience and is dense. This course explains the same foundational surface from the beginning, corrects version-sensitive examples for PyTorch 2.14, and then goes beyond it into compilation, distributed training, and export. A coverage matrix near the end maps every cheatsheet section to this course.

### Suggested pace

| Phase | Chapters | Typical time | Outcome |
|---|---:|---:|---|
| Foundations | 0–3 | 2–3 weeks | Think in shapes; derive and inspect gradients |
| Training systems | 4–7 | 3–4 weeks | Build reliable end-to-end training pipelines |
| Architectures | 8–10 | 3–4 weeks | Implement CNNs and a small transformer |
| Engineering | 11–15 | 3–4 weeks | Debug, profile, accelerate, compile, and export |
| Scale | 16–17 | 2–3 weeks | Use DDP/FSDP2 and advanced extension points |
| Capstone | 18 | 2–4 weeks | Deliver a reproducible project from empty folder |

Do not follow the calendar blindly. Advance when you meet the exit criteria.

### Course map

- **Part I:** Chapters 0–3 — learning loop, tensors, memory, autograd
- **Part II:** Chapters 4–7 — modules, data, training, reproducibility
- **Part III:** Chapters 8–10 — CNNs, attention, transformers
- **Part IV:** Chapters 11–17 — fine-tuning, debugging, AMP, compile, profiling, distributed, export
- **Part V:** Chapter 18 — capstone projects
- **Afterward:** retrieval drills, decision guide, official references, setup

---

# Part I — Foundations

---

# Part I — Foundations

## Chapter 0 — The machine-learning system before PyTorch

### What are we learning in this chapter?
- **The Core ML System Loop**: How machine learning actually works from scratch (predictions $\hat{y} = f(x; \theta)$, scalar loss $L$, gradient sensitivities $\nabla_\theta L$, and parameter updates).
- **The Five Fundamental PyTorch Objects**: Tensor, Module, Loss function, Optimizer, and DataLoader.
- **Derivatives as Local Sensitivities**: What a derivative really means in plain English (a "nudge" factor), derived step-by-step with real numbers.
- **Manual Gradient Descent Without Autograd**: Writing a complete training loop from scratch using basic tensor math (`linspace`, `randn_like`, `square`, `mean`) to see every moving part before PyTorch automates it.
- **Essential ML Vocabulary**: Samples vs. batches vs. epochs; parameters vs. hyperparameters; training vs. validation vs. test sets.

---

### 1. What problem does machine learning solve?

At its heart, machine learning is about finding numbers (called **parameters**) for a formula so that inputs produce the right outputs.

Imagine you want to predict a student's final exam score based on the number of hours they studied. The simplest relationship is a straight line:

$$\text{predicted score} = (w \times \text{hours}) + b$$

Here:
- `hours` is our input ($x$).
- `score` is the true answer ($y$).
- $w$ is the **weight** (how much each hour of study adds to the score).
- $b$ is the **bias** (the base score a student gets even with zero hours of study).
- $w$ and $b$ together are the **parameters** (often written as $\theta$).

Every machine learning system in the world—from this single line to a 70-billion-parameter language model—repeats the exact same 4-step loop:

1. **Predict:** Plug input $x$ into our formula to get a prediction $\hat{y}$ (read: "y-hat").
2. **Measure Error (Loss):** Compare our prediction $\hat{y}$ to the true answer $y$ using a single number called the **loss** $L(\hat{y}, y)$. The bigger the error, the bigger the loss.
3. **Calculate Sensitivity (Gradients):** Figure out how much each parameter ($w$ and $b$) contributed to that error. If we nudge $w$ up slightly, does the error go up or down?
4. **Update Parameters:** Nudge the parameters in the direction that makes the error smaller:
   $$\theta \leftarrow \theta - \eta \times (\text{gradient})$$
   where $\eta$ (eta) is the **learning rate**—a small step size we choose.

NumPy can easily do step 1. You could do step 3 with paper and pencil for small equations. PyTorch exists to automate step 3 (automatic differentiation), accelerate steps 1 and 4 on GPUs, and give you reusable Lego blocks for building complex models.

---

### 2. A derivative is just a "nudge factor", not magic

Many people find calculus intimidating. But in machine learning, you only need one core idea: **local sensitivity**.

Ask yourself this question:
> "If I increase the weight $w$ by a tiny amount, how much does the loss change?"

Let's walk through it with real numbers:
- Suppose a student studied for $x = 2$ hours.
- The true exam score was $y = 7$.
- Our current guess for the parameters is $w = 1.0$ and $b = 0.0$.
- Our prediction is $\hat{y} = (1.0 \times 2) + 0.0 = 2.0$.
- The error is $\hat{y} - y = 2.0 - 7.0 = -5.0$.

To measure error, we square it: $\text{Loss} = (-5.0)^2 = 25.0$. Why square it?
1. **Errors cannot cancel each other out:** If one prediction is $+5$ off and another is $-5$ off, we don't want their sum to be $0$! Squaring makes every error positive.
2. **Big mistakes get punished harder:** An error of $1$ squared is $1$, but an error of $10$ squared is $100$.

Now, let's see what happens if we nudge $w$ by a tiny amount $\delta$ (say, $\delta = 0.01$):
- The new weight is $w + \delta = 1.01$.
- The new prediction is $(1.01 \times 2) + 0.0 = 2.02$.
- The new error is $2.02 - 7.0 = -4.98$.
- The new loss is $(-4.98)^2 = 24.8004$.

Notice what happened: The loss changed from $25.0$ down to $24.8004$—a drop of $-0.1996$.
Divide the change in loss by the nudge size:
$$\frac{-0.1996}{0.01} \approx -20.0$$

That number, **$-20.0$**, is the **derivative**!
It tells you:
- The **sign** is negative: increasing $w$ *decreases* the loss.
- The **magnitude** is $20$: for every unit increase in $w$, the loss drops at a rate of 20 units.

Because the derivative is $-20$, if we subtract a fraction of it from $w$ (say, with learning rate $\eta = 0.05$):
$$w_{\text{new}} = w - (0.05 \times -20) = 1.0 + 1.0 = 2.0$$
Notice how $w$ moved from $1.0$ toward the correct slope!

---

### 3. The Five Core PyTorch Objects

As you learn PyTorch, you will see hundreds of functions, but they all serve five core objects:

```
+-------------------------------------------------------------------+
|                        1. torch.Tensor                            |
|  (Data container: numbers, shape, data type, device, gradients)   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       2. torch.nn.Module                          |
|    (The Model: holds parameters and defines forward computation)  |
+-------------------------------------------------------------------+
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
+-----------------------------+       +-----------------------------+
|    3. torch.utils.data      |       |      4. Loss Function       |
|         DataLoader          |       |   (Measures how bad our     |
|   (Feeds batches of data)   |       |      predictions are)       |
+-----------------------------+       +-----------------------------+
               |                                     |
               +------------------+------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       5. torch.optim.Optimizer                    |
|         (Updates model parameters based on their gradients)       |
+-------------------------------------------------------------------+
```

1. **Tensor:** A multi-dimensional grid of numbers (like a NumPy array) that can run on a GPU and remember how it was computed.
2. **Module:** A reusable neural network component (or the entire network) that stores weights and defines the forward pass.
3. **Loss:** A mathematical formula that takes `(prediction, target)` and outputs a single scalar penalty score.
4. **Optimizer:** The algorithm (like SGD or AdamW) that adjusts the weights using the gradients.
5. **DataLoader:** An automated conveyor belt that loads, shuffles, and groups data into batches.

---

### 4. Worked Example: Linear Regression from Scratch Without Autograd

Let's build a complete, runnable training loop using only basic PyTorch tensors and manual derivatives.

```python
import torch

# 1. Set a random seed so results are identical every time you run this
torch.manual_seed(0)

# 2. Create synthetic data: 100 points between -2 and +2
# True relationship: y = 3*x - 0.5 (with a tiny bit of random noise)
x = torch.linspace(-2, 2, 100)
y = 3 * x - 0.5 + 0.2 * torch.randn_like(x)

# 3. Initialize our parameters to 0.0
w = torch.tensor(0.0)
b = torch.tensor(0.0)
lr = 0.05  # Learning rate (step size)

# 4. The Training Loop: 200 steps
for step in range(200):
    # Step A: Predict
    pred = w * x + b
    
    # Step B: Measure error
    error = pred - y
    loss = (error.square()).mean()
    
    # Step C: Compute manual gradients
    # d(MSE)/dw = (2/N) * sum(error * x)
    # d(MSE)/db = (2/N) * sum(error)
    grad_w = (2 * error * x).mean()
    grad_b = (2 * error).mean()
    
    # Step D: Update parameters (move opposite to gradient)
    w -= lr * grad_w
    b -= lr * grad_b
    
    if (step + 1) % 50 == 0:
        print(f"Step {step+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.3f} | b: {b.item():.3f}")

# 5. Check that our parameters learned the true values (w=3.0, b=-0.5)
assert abs(w.item() - 3.0) < 0.1, f"Expected w ~ 3.0, got {w.item()}"
assert abs(b.item() - (-0.5)) < 0.1, f"Expected b ~ -0.5, got {b.item()}"
print("Success! The model learned the true line from scratch.")
```

---

### 5. Multivariate Regression: Scaling to Multiple Features with Matrices

In real life, an outcome rarely depends on just one input.
For example, predicting a house's price might depend on:
- $x_1$: Square footage
- $x_2$: Number of bedrooms

The prediction formula becomes:
$$\hat{y} = w_1 x_1 + w_2 x_2 + b$$

If we have $N = 500$ houses and $D = 2$ features, we don't write 500 separate equations. We organize the data into a **2D Matrix** $\mathbf{X}$ of shape `[500, 2]`, and our weights into a vector $\mathbf{w}$ of shape `[2]`:

#### A. Slicing Columns: `X[:, 0]`
In PyTorch, a 2D tensor has two axes: `[rows, columns]`.
- `:` means "select all rows".
- `0` means "select column 0".
So `X[:, 0]` extracts the entire first feature column (square footage) as a 1D tensor of shape `[500]`.

#### B. Matrix Multiplication: The `@` Operator
To compute predictions for all 500 houses in a single operation:
```python
# X has shape [500, 2], w has shape [2]
pred = X @ w + b  # Result has shape [500]
```
The `@` symbol is PyTorch's matrix multiplication operator. For each house, it multiplies each feature by its corresponding weight and sums them: $(x_{i, 1} \times w_1) + (x_{i, 2} \times w_2) + b$.

#### C. Matrix Transpose: `X.t()`
When computing the gradient for the weights, each feature's sensitivity depends on the error across all 500 samples:
$$\text{grad\_w}_j = \frac{2}{N} \sum_{i=1}^N x_{i, j} \cdot (\hat{y}_i - y_i)$$

In matrix form, multiplying the transpose of $\mathbf{X}$ (shape `[2, 500]`) by the error vector (shape `[500]`) computes this exact sum for all weights simultaneously:
```python
error = pred - y  # Shape [500]
grad_w = (2.0 / 500) * (X.t() @ error)  # Shape [2]
grad_b = (2 * error).mean()             # Scalar
```
Here, `X.t()` flips the rows and columns of $\mathbf{X}$ so the dimensions match for matrix multiplication (`[2, 500] @ [500] -> [2]`).

#### D. Tensor Verification: `.item()` and `torch.allclose()`
- **`tensor.item()`**: When a tensor contains only a single scalar number (like bias `b` or `loss`), `.item()` extracts it as a native Python `float`.
- **`torch.allclose(a, b, atol=...)`**: Because floating point arithmetic contains tiny rounding errors (e.g. `2.0000001` vs `2.0`), never use `a == b` for floats! Instead, use `torch.allclose(a, b, atol=0.1)` to check if all values in two tensors are equal within a specified tolerance.

---

### Assignments 0

#### Assignment 0.1 — Manual MSE Gradient Derivation
- **Objective**: Derive and verify the manual gradient formulas for simple linear regression under Mean Squared Error (MSE).
- **Given**:
  - Model: $\hat{y}_i = w \cdot x_i + b$
  - Loss function: $L = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$
- **Build / Do**:
  1. Differentiate a single squared error term with respect to $w$: $\frac{\partial}{\partial w} (\hat{y}_i - y_i)^2 = 2(\hat{y}_i - y_i)x_i$.
  2. Differentiate a single squared error term with respect to $b$: $\frac{\partial}{\partial b} (\hat{y}_i - y_i)^2 = 2(\hat{y}_i - y_i)$.
  3. Apply linearity of differentiation to the full mean across $N$ samples.
- **Pass Criteria**:
  - Your written derivation explicitly shows why the factor $\frac{2}{N}$ appears in both $\frac{\partial L}{\partial w} = \frac{2}{N}\sum_i(\hat{y}_i-y_i)x_i$ and $\frac{\partial L}{\partial b} = \frac{2}{N}\sum_i(\hat{y}_i-y_i)$.
  - Verify that the analytical formula matches the tensor code: `(2 * error * x).mean()` and `(2 * error).mean()`.

#### Assignment 0.2 — Learning Rate Dynamics Experiment
- **Objective**: Observe how learning rate magnitude affects optimization dynamics (slow convergence, stable convergence, oscillation, and divergence).
- **Given**:
  - Synthetic dataset:
    ```python
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 100)
    y = 3 * x - 0.5 + 0.2 * torch.randn_like(x)
    ```
  - Initial parameters: `w = torch.tensor(0.0)`, `b = torch.tensor(0.0)`.
- **Build / Do**:
  1. Run 200 optimization iterations separately for four learning rates: $\eta \in \{0.001, 0.1, 1.0, 10.0\}$.
  2. In each run, record the loss every 10 steps.
  3. Characterize each run based on the empirical trajectory:
     - $\eta = 0.001$: Slow, steady convergence (understepping).
     - $\eta = 0.1$: Fast, stable convergence to $w \approx 3, b \approx -0.5$.
     - $\eta = 1.0$: Severe oscillation across the loss minimum.
     - $\eta = 10.0$: Numerical explosion / divergence ($L \to \infty$ or `NaN`).
- **Pass Criteria**:
  - Record the step-loss curves for all four rates.
  - Successfully demonstrate and explain the transition from slow convergence to divergence using empirical numbers.

#### Assignment 0.3 — Multivariate Linear Regression with Manual Gradients
- **Objective**: Extend manual differentiation and update rules from scalar input to a 2D feature matrix using matrix operations.
- **Given**:
  - Seed: `torch.manual_seed(0)`
  - Input matrix: `X = torch.randn(500, 2)` (shape `[500, 2]`)
  - Target vector: `y = 2 * X[:, 0] - 4 * X[:, 1] + 1.0 + 0.05 * torch.randn(500)` (shape `[500]`)
  - Initial weights: `w = torch.zeros(2)` (shape `[2]`), scalar bias `b = torch.tensor(0.0)`
  - Learning rate: $\eta = 0.05$, total steps: 300
- **Build / Do**:
  1. Forward pass: `pred = X @ w + b` (shape `[500]`).
  2. Compute error: `error = pred - y`.
  3. Compute manual gradients:
     - Weight gradient: `grad_w = (2.0 / 500) * (X.t() @ error)` (or `(2 * error[:, None] * X).mean(dim=0)`).
     - Bias gradient: `grad_b = (2 * error).mean()`.
  4. In-place parameter updates: `w -= lr * grad_w`, `b -= lr * grad_b`.
- **Pass Criteria**:
  - Assert that learned weights are within `0.1` of true coefficients: `assert torch.allclose(w, torch.tensor([2.0, -4.0]), atol=0.1)`.
  - Assert that learned bias is within `0.1` of true bias: `assert abs(b.item() - 1.0) < 0.1`.

#### Assignment 0.4 — Closed-Book Retrieval
- **Objective**: Reinforce core concepts by recreating the foundational ML training loop without consulting notes or references.
- **Given**: An empty Python file.
- **Build / Do**:
  - From memory, write the complete script: data generation with noise, parameter initialization, forward prediction, MSE loss, analytical gradient calculations, and parameter update loop for 200 steps.
- **Pass Criteria**:
  - Script executes cleanly from a fresh process and converges to the underlying parameters ($w \approx 3, b \approx -0.5$).

**Exit criterion:** You can explain the 4-step loop (predict, loss, gradient, update) and why each step exists.

---

## Chapter 1 — Tensors from zero: values, axes, dtype, and device

### What are we learning in this chapter?
- **Why Tensors Exist**: Why Python lists are too slow, memory-inefficient, and incapable of GPU acceleration or automatic differentiation.
- **Dimensionality (Rank & Axes)**: Understanding 0D scalars, 1D vectors, 2D matrices, 3D sequence tensors, and 4D image batches from the ground up.
- **Tensor Creation by Intent**: Choosing the right constructor (`zeros`, `ones`, `full`, `empty`, `rand`, `randn`, `arange`, `linspace`, `eye`, `tensor`, `as_tensor`, `from_numpy`, `*_like`).
- **Data Types (Dtypes)**: How numbers are stored in memory (`float32`, `float64`, `float16`, `bfloat16`, `int64`, `bool`) and why precision vs. range matters.
- **Logical Devices**: Moving tensors between CPU and GPU (`torch.device`, `torch.accelerator`, `.to(device)`).
- **Randomness & Generators**: Using `torch.manual_seed` and independent `torch.Generator` instances for reproducible experiments.
- **Multi-Modal Dataset Auditing**: Constructing multi-dimensional tensors, measuring memory footprints with `numel()` and `element_size()`, and verifying device placement.

---

### 1. What problem does a tensor solve?

Suppose you want to store a 28x28 grayscale image of a handwritten digit. In plain Python, you could create a list of 28 lists, where each inner list has 28 numbers:

```python
# A 28x28 image stored as a Python list of lists
image = [[0 for _ in range(28)] for _ in range(28)]
```

This works for a toy script, but it immediately creates four massive problems for real machine learning:
1. **Memory Inefficiency:** A Python list doesn't store raw numbers side-by-side. It stores a list of *pointers* to separate Python integer/float objects scattered across your RAM. A 100MB dataset in Python lists can eat several gigabytes of memory!
2. **Slow Computations:** Because numbers are scattered, the computer's CPU cache cannot pre-fetch them efficiently. Operating on them requires slow Python interpreter loops instead of fast C/CUDA vectorized instructions.
3. **No GPU Acceleration:** A graphics card (GPU) contains thousands of tiny cores that can perform millions of multiplications in parallel—but only if the numbers are laid out in a single, contiguous block of memory with a known data type. Python lists cannot live on a GPU.
4. **No Gradient Tracking:** Python lists have no memory of how they were computed. They cannot automatically compute derivatives.

A `torch.Tensor` solves every one of these problems.
Think of a tensor as:
> **A contiguous block of numbers in memory, wrapped with a label that describes its shape, data type (dtype), hardware location (device), and optional gradient history.**

---

### 2. The Mental Model: Thinking in Shapes and Axes

The **rank** (or `ndim`) of a tensor is simply the number of axes (dimensions) it has.

Let's build up from zero:
- **0D Tensor (Scalar):** A single number with zero axes.
  ```python
  scalar = torch.tensor(5.0)  # shape: []
  ```
  *Mental model:* A single temperature reading: `24.5`.

- **1D Tensor (Vector):** A list of numbers with 1 axis.
  ```python
  vector = torch.tensor([5.0, 7.0, 9.0])  # shape: [3]
  ```
  *Mental model:* A shopping list or a series of hourly temperatures: `[20.1, 21.4, 22.0]`.

- **2D Tensor (Matrix):** A grid of numbers with 2 axes (rows and columns).
  ```python
  matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape: [2, 2]
  ```
  *Mental model:* A spreadsheet table. Rows are samples (e.g. 100 students), columns are features (e.g. height, weight, test score): `[100, 3]`.

- **3D Tensor:** A sequence or cube of numbers with 3 axes.
  *Mental model:* Sentences of text. Shape `[B, T, D]` = `[Batch Size, Sequence Length, Feature/Embedding Dim]`. For example, a batch of 32 sentences, where each sentence has 50 words, and each word is represented by a 128-dimensional embedding vector: `[32, 50, 128]`.

- **4D Tensor:** A collection of grids with 4 axes.
  *Mental model:* A batch of images. Shape `[B, C, H, W]` = `[Batch Size, Channels, Height, Width]`. For example, 64 color images of size 32x32: `[64, 3, 32, 32]`.

```
0D: Scalar           1D: Vector               2D: Matrix (Table)
  [ 5 ]              [ 1, 2, 3 ]              +---+---+---+
                                              | 1 | 2 | 3 |
                                              +---+---+---+
                                              | 4 | 5 | 6 |
                                              +---+---+---+

3D: Sequence (Text)                           4D: Image Batch
  [Batch, Tokens, Features]                     [Batch, Channels, Height, Width]
      +---+---+                                       +---+---+
     /   /   /|                                      /   /   /|
    +---+---+ +                                     +---+---+ +  (Channels: RGB)
    |   |   |/                                      |   |   |/   (Height x Width)
    +---+---+                                       +---+---+
   (Tokens x Features stacked per batch)           (Stacked for N images in batch)
```

---

### 3. Creating Tensors: Choose by Intent

Never memorize random tensor creation functions. Always ask: **"What do I need this tensor for?"**

#### A. When you need known fill values
```python
zeros = torch.zeros(2, 3)          # Fill with 0.0 (safe default)
ones = torch.ones(2, 3)            # Fill with 1.0
sevens = torch.full((2, 3), 7.0)   # Fill with any custom number
uninit = torch.empty(2, 3)         # Uninitialized memory! (DO NOT use unless immediately overwriting)
```
> [!WARNING]
> `torch.empty` does NOT mean "empty list". It allocates memory without clearing it, so it contains random leftover garbage from other programs. If you don't overwrite every value, you will get silent bugs!

#### B. When you need random numbers
```python
uniform = torch.rand(2, 3)              # Uniform distribution in [0.0, 1.0)
normal = torch.randn(2, 3)              # Standard normal distribution (bell curve, mean=0, std=1)
integers = torch.randint(0, 10, (2, 3)) # Random integers from 0 up to 9
perm = torch.randperm(5)                # Shuffled integers: e.g. [3, 0, 4, 1, 2]
```
> [!NOTE]
> `rand` and `randn` are completely different!
> - `torch.rand`: Bounded strictly between $0$ and $1$. Mean is $0.5$.
> - `torch.randn`: Unbounded bell curve. Mean is $0.0$, standard deviation is $1.0$.

#### C. When you need regular sequences of numbers
```python
# Use arange when you know the STEP size:
steps = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8] (stop 10 is excluded)

# Use linspace when you know the COUNT of points and want both endpoints:
points = torch.linspace(0, 1, 5)  # [0.00, 0.25, 0.50, 0.75, 1.00]
```

#### D. When converting from Python or NumPy
```python
# 1. torch.tensor(data): Always COPIES data (safe, but uses extra memory)
a = torch.tensor([1, 2, 3])

# 2. torch.as_tensor(data): Avoids copying when possible (shares memory with NumPy)
b = torch.as_tensor(numpy_array)

# 3. clone().detach(): The safest way to duplicate an existing PyTorch tensor
c = existing_tensor.clone().detach()
```

#### E. When matching an existing tensor's shape, dtype, and device
```python
x = torch.randn(4, 5, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)

# *_like functions automatically copy the shape, dtype, and device of x:
y = torch.zeros_like(x)
z = torch.randn_like(x)
```

#### F. When you need structured matrices (Identity and Diagonal)
```python
# 1. torch.eye(n): Creates an n x n 2D identity matrix (1s on the diagonal, 0s elsewhere)
identity = torch.eye(4)  # Shape: [4, 4]

# 2. t.diag(): Extracts the main diagonal of a 2D matrix as a 1D tensor
diag_vals = identity.diag()  # Shape: [4], values: [1.0, 1.0, 1.0, 1.0]
```

---

### 4. Dtype and Device: How and Where Numbers Live

#### Data Types (Dtypes)
A **dtype** tells PyTorch how many bits to use for each number:
- `torch.float32` (Default): 32-bit floating point. Standard for deep learning weights and activations.
- `torch.float64` (Double): 64-bit float. Used for high-precision scientific computing or checking gradient accuracy.
- `torch.float16` (Half): 16-bit float. Half the memory of float32, faster on GPUs, but small range (can overflow if numbers > 65,504).
- `torch.bfloat16` (Brain Float): 16-bit float with the same range as float32, but less precision. Highly popular on modern GPUs.
- `torch.int64` / `torch.long`: 64-bit integer. **Required** for classification labels, token IDs, and indices.
- `torch.bool`: Boolean (`True`/`False`). Used for masks.

#### Converting Dtypes (Casting)
To convert an existing tensor from one dtype to another, use `.to(dtype)` or the built-in shorthand methods:
```python
x = torch.ones(5)                # Default: float32 (4 bytes per element)

x_f64 = x.to(torch.float64)      # Shorthand: x.double()
x_f16 = x.to(torch.float16)      # Shorthand: x.half()
x_bf16 = x.to(torch.bfloat16)    # Shorthand: x.bfloat16()
x_i64 = x.to(torch.int64)        # Shorthand: x.long()
x_bool = x.to(torch.bool)        # Shorthand: x.bool()
```

#### Measuring Memory: `numel()`, `element_size()`, and Storage
How much RAM does your tensor actually consume?
1. **`t.numel()`**: Returns the total count of numbers in the tensor (e.g., shape `[2, 5]` has `10` elements).
2. **`t.element_size()`**: Returns the number of bytes used by a single element (e.g., `8` for `float64`, `4` for `float32`, `2` for `float16`, `1` for `bool`).
3. **`t.untyped_storage().nbytes()`**: Directly queries the underlying memory buffer allocated in RAM to return the actual total byte count.
```python
t = torch.ones(1_000_000, dtype=torch.float32)
theoretical_bytes = t.numel() * t.element_size()  # 1,000,000 * 4 = 4,000,000 bytes (~4 MB)
actual_bytes = t.untyped_storage().nbytes()       # 4,000,000 bytes
```

#### Devices: CPU vs. Accelerator (GPU)
Your computer has two distinct memory spaces:
1. **Host Memory (RAM):** Used by your CPU.
2. **Device Memory (VRAM):** Dedicated high-speed memory on your GPU.

Tensors cannot interact across devices. If you try to add a CPU tensor to a GPU tensor, PyTorch will throw an error:
```python
# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(3, 3)          # Lives on CPU
x = x.to(device)               # Moved to GPU (if available)
y = torch.ones_like(x)         # Automatically created on GPU to match x
z = x + y                      # Valid: both live on the same device!
```

---

### 5. Randomness, Extraction, and Tensor Health

#### A. Isolated Random Streams with `torch.Generator`
By default, `torch.manual_seed(0)` sets the **global** seed for the entire PyTorch process.
If multiple parts of your code generate random numbers (e.g. data augmentation vs weight initialization), they will advance the same global counter.
To keep random streams completely independent and reproducible, create dedicated `torch.Generator` instances:
```python
g1 = torch.Generator()
g1.manual_seed(123)

g2 = torch.Generator()
g2.manual_seed(123)

# Both generators start at seed 123, producing identical sequences:
t1 = torch.randn(3, generator=g1)
t2 = torch.randn(3, generator=g2)
assert torch.equal(t1, t2)
```

#### B. Extracting Values: `.item()`, `.tolist()`, `.min()`, and `.max()`
- **`t.item()`**: Unpacks a 0D tensor or 1-element tensor into a native Python `float` or `int`. Essential when logging loss or comparing with float thresholds. (Calling `.item()` on a tensor with more than 1 element raises `RuntimeError`.)
- **`t.tolist()`**: Converts a tensor of any rank into a nested standard Python list (useful for serializing to JSON or checking permutations with Python `set`).
- **`t.min()` and `t.max()`**: Find the minimum and maximum values across the entire tensor, returned as 0D scalar tensors.

```python
# 1. Unpacking a single scalar with .item():
scalar_tensor = torch.tensor(3.14159)
py_float = scalar_tensor.item()
assert isinstance(py_float, float)
assert round(py_float, 4) == 3.1416

# Also works on 1-element tensors of any rank (e.g., shape [1, 1]):
single_elem = torch.tensor([[42]])
assert single_elem.item() == 42
assert isinstance(single_elem.item(), int)

# 2. Converting tensors to Python lists with .tolist():
grid = torch.tensor([[1, 2, 3], [4, 5, 6]])
nested_list = grid.tolist()
assert nested_list == [[1, 2, 3], [4, 5, 6]]
assert isinstance(nested_list, list)

# 3. Finding min and max across a tensor:
# On integer tensors:
counts = torch.tensor([15, -4, 99, 0])
assert counts.min().item() == -4
assert counts.max().item() == 99

# On float tensors (note: float32 precision when unpacking to Python float):
measurements = torch.tensor([15.2, -4.8, 99.1, 0.0])
min_val = measurements.min()  # Returns a 0D tensor: tensor(-4.8000)
max_val = measurements.max()  # Returns a 0D tensor: tensor(99.1000)

assert round(min_val.item(), 1) == -4.8
assert round(max_val.item(), 1) == 99.1
```

#### C. Testing Equality and Boolean Reductions: `torch.equal()` and `.all()`
- In PyTorch, `t == 0` does **not** return a single `True` or `False`! It returns a tensor of booleans matching the shape of `t`.
- To check if **every** element meets the condition, call `.all()`:
  ```python
  t = torch.zeros(3, 4)
  assert (t == 0.0).all()  # True only if every single element is 0.0
  ```
- To test if two tensors have identical shapes and identical values in one call, use **`torch.equal(a, b)`**:
  ```python
  assert torch.equal(torch.zeros(2, 2), torch.zeros(2, 2))
  ```

#### D. Numerical Health: Detecting `NaN` and `Inf` with `torch.isfinite()`
When a calculation divides by zero or overflows, it produces `inf` (infinity) or `nan` (not a number).
Use `torch.isfinite(t)` to check that all values are valid numbers:
```python
x = torch.tensor([1.0, 2.0, 3.0])
assert torch.isfinite(x).all()  # True

corrupt = torch.tensor([1.0, float('nan'), 3.0])
assert not torch.isfinite(corrupt).all()  # False! Caught NaN before it spreads.
```

---

### 6. Worked Example: Constructing a Multi-Modal Dataset & Auditing Memory

Let's combine all of Chapter 1's lessons into a real-world task: constructing synthetic training data for a multi-modal model (images + tabular features + class labels), casting data types to cut memory consumption in half, and moving them to the accelerator:

```python
import torch

# 1. Ensure reproducibility with an isolated generator:
g = torch.Generator()
g.manual_seed(42)

# 2. Tabular features: 1,000 samples with 4 continuous features:
X_tab = torch.randn(1000, 4, generator=g, dtype=torch.float32)

# 3. Categorical labels: 1,000 integer class IDs (0, 1, 2):
y_tab = torch.randint(0, 3, (1000,), generator=g, dtype=torch.int64)

# 4. Image batch: 16 color images (3 channels, 32x32 pixels):
images = torch.rand(16, 3, 32, 32, generator=g, dtype=torch.float32)

# 5. Dtype downcasting to save memory:
# Float32 uses 4 bytes per number. Float16 uses only 2 bytes!
images_fp16 = images.half()

# 6. Memory Audit:
# Theoretical bytes = numel() * element_size()
images_fp32_bytes = images.numel() * images.element_size()          # 16*3*32*32 * 4 = 196,608 bytes
images_fp16_bytes = images_fp16.numel() * images_fp16.element_size() # 16*3*32*32 * 2 = 98,304 bytes

assert images_fp32_bytes == images.untyped_storage().nbytes()
assert images_fp16_bytes == images_fp16.untyped_storage().nbytes()
assert images_fp16_bytes == images_fp32_bytes / 2  # Exactly 50% memory saved!

# 7. Move to target hardware:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tab = X_tab.to(device)
y_tab = y_tab.to(device)
images_fp16 = images_fp16.to(device)

# 8. Sanity check dataset health:
assert torch.isfinite(X_tab).all()
assert torch.isfinite(images_fp16).all()
assert X_tab.device == y_tab.device == images_fp16.device

print(f"Dataset successfully created and audited on {device}!")
print(f"Images FP32 memory: {images_fp32_bytes / 1024:.1f} KB")
print(f"Images FP16 memory: {images_fp16_bytes / 1024:.1f} KB (50% reduction)")
```

---

### Assignments 1

#### Assignment 1.1 — Creation Laboratory
- **Objective**: Master tensor creation functions and choose the appropriate constructor based on semantic intent.
- **Given**: No input data.
- **Build / Do**:
  1. Create a $3 \times 4$ matrix filled with zeros.
  2. Create a $2 \times 5$ matrix filled with float `-3.0`.
  3. Create an integer sequence containing values `5, 10, 15, 20, 25` using `torch.arange`.
  4. Create a 1D tensor of eleven evenly spaced floats from `-1.0` to `1.0` inclusive using `torch.linspace`.
  5. Create a $4 \times 4$ identity matrix using `torch.eye`.
  6. Create a random permutation of integers $0$ through $9$ using `torch.randperm`.
- **Pass Criteria**:
  - Check every tensor with `assert` for exact shape, dtype, and boundary values:
    - Zeros: `shape == (3, 4)`, `dtype == torch.float32`, `(t == 0).all()`.
    - `-3.0` matrix: `shape == (2, 5)`, `(t == -3.0).all()`.
    - Arange: `shape == (5,)`, `dtype == torch.int64`, `t[0] == 5`, `t[-1] == 25`.
    - Linspace: `shape == (11,)`, `t[0] == -1.0`, `t[-1] == 1.0`, `t[5] == 0.0`.
    - Identity: `shape == (4, 4)`, `torch.equal(t.diag(), torch.ones(4))`.
    - Permutation: `shape == (10,)`, `set(t.tolist()) == set(range(10))`.
  - Provide written explanation of when to use `arange` (known step) vs. `linspace` (known endpoint count).

#### Assignment 1.2 — Empirical Distribution Experiment
- **Objective**: Verify the empirical statistical properties and bounds of uniform and normal random distributions.
- **Given**: Two synthetic sample tensors of size $1,000,000$:
  - Uniform: `u = torch.rand(1_000_000)`
  - Normal: `n = torch.randn(1_000_000)`
- **Build / Do**:
  1. Compute the empirical mean, standard deviation, minimum, and maximum for both tensors.
  2. Compare observed values against theoretical expectations:
     - Uniform $\mathcal{U}[0, 1)$: $\mu = 0.5$, $\sigma = \frac{1}{\sqrt{12}} \approx 0.2887$, bounded in $[0, 1)$.
     - Normal $\mathcal{N}(0, 1)$: $\mu = 0.0$, $\sigma = 1.0$, unbounded (min/max typically between $\pm 4.5$ and $\pm 5.5$).
- **Pass Criteria**:
  - Assert that `abs(u.mean().item() - 0.5) < 0.01` and `abs(u.std().item() - 0.2887) < 0.01`.
  - Assert that `u.min() >= 0.0` and `u.max() < 1.0`.
  - Assert that `abs(n.mean().item()) < 0.01` and `abs(n.std().item() - 1.0) < 0.01`.
  - Written explanation of why `rand` and `randn` cannot be used interchangeably.

#### Assignment 1.3 — Dtype Precision and Memory Footprint
- **Objective**: Quantify the memory footprint and numerical properties of different tensor data types.
- **Given**: A 1D tensor of ones: `x = torch.ones(1_000_000)`.
- **Build / Do**:
  1. Convert `x` to `torch.float64`, `torch.float16`, `torch.bfloat16`, and `torch.bool`.
  2. For each tensor, calculate theoretical bytes: `t.numel() * t.element_size()`.
  3. Verify against actual storage bytes using `t.untyped_storage().nbytes()`.
  4. Tabulate: Dtype, bytes per element, total megabytes, and valid value range / precision.
- **Pass Criteria**:
  - Table accurately lists: `float64` (8 bytes, ~8 MB), `float32` (4 bytes, ~4 MB), `float16` (2 bytes, ~2 MB), `bfloat16` (2 bytes, ~2 MB), `bool` (1 byte, ~1 MB).
  - Written explanation explaining why `float16`/`bfloat16` save memory but require care with dynamic range (underflow/overflow), and why `bool` is unsuitable for continuous weights.

#### Assignment 1.4 — Independent Random Streams with `torch.Generator`
- **Objective**: Ensure reproducibility and isolation between different stochastic components using explicit generator state.
- **Given**: Two distinct `torch.Generator` instances: `g1 = torch.Generator()` and `g2 = torch.Generator()`.
- **Build / Do**:
  1. Seed both generators with `123`: `g1.manual_seed(123)` and `g2.manual_seed(123)`.
  2. Draw $t_{1a} = \text{randn}(3, \text{generator}=g_1)$ and $t_{2a} = \text{randn}(3, \text{generator}=g_2)$.
  3. Draw $t_{1b} = \text{randn}(3, \text{generator}=g_1)$ and $t_{2b} = \text{randn}(3, \text{generator}=g_2)$.
- **Pass Criteria**:
  - `assert torch.equal(t1a, t2a)` (generators with the same seed produce identical initial sequences).
  - `assert torch.equal(t1b, t2b)` (subsequent draws remain synchronized).
  - `assert not torch.equal(t1a, t1b)` (successive draws from the same stream advance state and differ).
  - `assert not torch.equal(t2a, t2b)`.

#### Assignment 1.5 — Multi-Modal Memory Budget & Device Placement Audit
- **Objective**: Construct a multi-modal dataset pipeline, quantify memory savings through dtype downcasting, and verify device placement.
- **Given**:
  - Tabular features: 5,000 samples with 20 features.
  - Class labels: 5,000 integer labels in $\{0, 1, 2, 3\}$.
  - Image batch: 64 color images of shape `[64, 3, 32, 32]`.
- **Build / Do**:
  1. Construct tabular features with `torch.randn` as `float32`.
  2. Construct labels with `torch.randint` as `int64`.
  3. Construct images with `torch.rand` as `float32`.
  4. Downcast images to `float16` (`.half()`) and tabular features to `float64` (`.double()`).
  5. Compute theoretical bytes (`t.numel() * t.element_size()`) for all tensors and assert exact match with `t.untyped_storage().nbytes()`.
  6. Move all tensors to target device (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
  7. Verify with assertions: all tensors reside on `device`, dtypes are exact, and `torch.isfinite(t).all()` holds.
- **Pass Criteria**:
  - Memory audit asserts:
    - Images (FP32): `64 * 3 * 32 * 32 * 4 == 786432` bytes (~786 KB).
    - Images (FP16): `64 * 3 * 32 * 32 * 2 == 393216` bytes (~393 KB).
    - Tabular (FP64): `5000 * 20 * 8 == 800000` bytes (~800 KB).
    - Labels (INT64): `5000 * 8 == 40000` bytes (~40 KB).
  - All assertions pass without error.

**Exit criterion:** You can choose the correct tensor creation function based on intent, cast dtypes, audit memory footprints, and manage device placement without guessing.

---

## Chapter 2 — Tensor mechanics: shapes, indexing, broadcasting, and memory

### What are we learning in this chapter?
- **Shape Transformations Demystified**: What `reshape`, `flatten`, `transpose`, `permute`, `squeeze`, and `unsqueeze` actually do to your data.
- **Combining & Splitting Tensors**: The crucial difference between `torch.cat` (gluing along an existing axis) and `torch.stack` (creating a new axis).
- **Reductions in Plain English**: Understanding what the `dim` parameter actually means (the axes that get squashed), `keepdim=True`, `std`/`var` Bessel's correction, and numerical clamping (`clamp_min`).
- **Broadcasting from First Principles**: How PyTorch stretches tensors of different shapes without copying memory, and how to avoid the deadly `[B, 1] - [B] -> [B, B]` loss bug.
- **Indexing: Views vs. Copies**: Why basic slice indexing modifies the original tensor (view) while integer/boolean indexing creates a new copy.
- **Advanced Selection with `gather` and `scatter`**: The "address book" and "sorting mail" mental models for reading and writing tensor values.
- **Einstein Summation (`torch.einsum`)**: How to write clean, multi-dimensional matrix operations in a single readable equation.
- **Physical Memory Under the Hood**: Strides, storage offsets, and contiguous memory layouts.
- **Sliding Windows (`unfold`) & Testing Tools**: Efficient windowing without loops and robust assertion testing with `torch.testing.assert_close`.
- **Worked Example: Image Batch Normalization**: Standardizing multi-channel image batches using reductions, `keepdim=True`, broadcasting, and numerical clamping.

---

### 1. Shape Operations: Regrouping vs. Reordering

When working with neural networks, 80% of your code involves reshaping tensors.
Different operations do fundamentally different things:

```python
x = torch.arange(24).reshape(2, 3, 4)
```

1. **Regrouping (Changing grid dimensions without moving items in memory):**
   - `x.reshape(6, 4)`: Takes the same 24 numbers in reading order (left-to-right, top-to-bottom) and groups them into 6 rows of 4.
   - `x.flatten(1)`: Flattens dimensions from axis 1 onward. Shape becomes `[2, 12]`.

2. **Reordering (Swapping or shuffling the axes):**
   - `x.transpose(1, 2)`: Swaps axis 1 and axis 2. Shape becomes `[2, 4, 3]`.
   - `x.permute(2, 0, 1)`: Completely reorganizes all axes in a new custom order. Shape becomes `[4, 2, 3]`.

3. **Inserting or Removing Size-1 Dimensions:**
   - `x.unsqueeze(1)`: Inserts a new dimension of size 1 at position 1. Shape becomes `[2, 1, 3, 4]`.
   - `x.squeeze(1)`: Removes dimension 1 *only if its size is 1*.

> [!CAUTION]
> **The Bare `squeeze()` Trap:**
> If you write `x.squeeze()`, PyTorch removes *all* dimensions of size 1.
> If your batch size happens to be 1 (e.g. `[1, 10]`), bare `squeeze()` will turn it into `[10]`, accidentally stripping away your batch dimension and breaking downstream layers!
> **Rule:** Always specify the exact dimension: `x.squeeze(dim)`.

---

### 2. Combining and Splitting: `cat` vs. `stack`

This is one of the most common beginner confusions. The difference is simple:

- **`torch.cat` (Concatenate):** Glues tensors along an **existing** axis. No new dimension is created.
  - Think of taping two sheets of paper side-by-side.
  - Inputs: Two `[2, 3]` tensors $\implies$ `torch.cat([a, b], dim=0)` produces `[4, 3]`.
- **`torch.stack`:** Stacks tensors along a **new** axis.
  - Think of stacking two sheets of paper on top of each other into a two-page booklet.
  - Inputs: Two `[2, 3]` tensors $\implies$ `torch.stack([a, b], dim=0)` produces `[2, 2, 3]`.

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

cat_rows = torch.cat([a, b], dim=0)   # Shape: [4, 3] (Rows joined)
cat_cols = torch.cat([a, b], dim=1)   # Shape: [2, 6] (Columns joined)
stacked  = torch.stack([a, b], dim=0) # Shape: [2, 2, 3] (Brand new dimension!)
```

---

### 3. Reductions: What does `dim` mean?

When you call `.mean(dim=...)` or `.sum(dim=...)`, ask yourself:
> **"Which dimension do I want to squash/eliminate?"**

Suppose you have an image batch of shape `[8, 3, 32, 32]` (Batch, Channel, Height, Width):
- `x.mean(dim=0)` $\implies$ Squashes the batch dimension $\implies$ Shape `[3, 32, 32]` (average image).
- `x.mean(dim=(2, 3))` $\implies$ Squashes height and width $\implies$ Shape `[8, 3]` (average color per image).
- `x.mean(dim=(2, 3), keepdim=True)` $\implies$ Keeps squashed dimensions as size 1 $\implies$ Shape `[8, 3, 1, 1]`.

`keepdim=True` is vital because `[8, 3, 1, 1]` can broadcast back against `[8, 3, 32, 32]` automatically!

#### Standard Deviation (`std`) and Bessel's Correction (`correction`)
When computing spread with `x.std(dim=...)` or variance `x.var(dim=...)`:
- **`correction=1` (Sample Standard Deviation — Default):** Divides by $N - 1$. Used in classical statistics when estimating the standard deviation of an entire population from a small sample.
- **`correction=0` (Population Standard Deviation):** Divides by $N$. **This is standard in deep learning normalization layers** (such as BatchNorm and LayerNorm) because the current batch is treated as the complete population being normalized.

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
# Sample std (divides by N - 1 = 4):
sample_std = x.std(correction=1)  # 1.5811
# Population std (divides by N = 5):
pop_std = x.std(correction=0)     # 1.4142
```

#### Guarding Against Zero Division with Clamping (`clamp_min`)
When normalizing data $(\frac{x - \mu}{\sigma})$, what happens if all pixel values in an image are identical (e.g., a solid black frame)?
The standard deviation $\sigma$ will be exactly `0.0`. Dividing by zero produces `NaN` or `Inf`, corrupting training!
To prevent this, PyTorch provides clamping:
- **`x.clamp_min(min_val)`:** Replaces any value smaller than `min_val` with `min_val`.
- **`torch.clamp(x, min=min_val, max=max_val)`:** Restricts all values into the range `[min_val, max_val]`.

```python
std = torch.tensor([0.0, 0.5, 1.2])
# Guard against zero with an epsilon:
safe_std = std.clamp_min(1e-6)  # [1e-6, 0.5, 1.2]
```

---

### 4. Broadcasting: The 3-Step Alignment Rule

Broadcasting allows you to add, subtract, or multiply tensors of different shapes without copying data.

**The 3-Step Checklist:**
1. **Right-align** the shapes on paper.
2. For each dimension starting from the **rightmost**:
   - Are the numbers equal? $\implies$ Valid!
   - Is one of them `1`? $\implies$ Valid! (The size 1 stretches to match the larger number).
   - Is one of them missing? $\implies$ Valid! (Treated as size 1 and stretches).
   - Are the numbers different and neither is `1`? $\implies$ **Error! Incompatible shapes.**

#### Example:
```
Tensor A:     [5,  3,  1]
Tensor B: [8,  1,  3,  4]
-------------------------
Result:   [8,  5,  3,  4]  (Valid!)
```
From right to left:
- Axis 0 (right): `1` and `4` $\implies$ stretches to `4`.
- Axis 1: `3` and `3` $\implies$ matches `3`.
- Axis 2: `5` and `1` $\implies$ stretches to `5`.
- Axis 3 (left): Missing and `8` $\implies$ stretches to `8`.

#### The Deadly Silent Bug: `[B, 1] - [B] -> [B, B]`
If your model outputs predictions `pred` of shape `[32, 1]` and your targets `target` are 1D of shape `[32]`:
```
pred:   [32,  1]
target: [    32]  (aligned to right!)
----------------
Result: [32, 32]  <-- DISASTER!
```
PyTorch will NOT throw an error. It will broadcast your 32 predictions against all 32 targets, creating an unintended $32 \times 32$ matrix! The loss will be calculated over 1,024 pairs instead of 32.
**Fix:** Always ensure dimensions match before loss: `target.unsqueeze(1)` or `assert pred.shape == target.shape`.

---

### 5. Indexing: Views vs. Copies

- **Basic Indexing (Slices, integers, `None`, `...`):**
  Returns a **VIEW**. It shares the exact same memory. If you change the slice, you change the original tensor!
  - Slicing: `x[0:2, :]`
  - Ellipsis (`...`): Replaces multiple `:` for all intermediate dimensions. For example, for a 4D tensor `[2, 3, 4, 5]`, `x[..., -1]` grabs the last element along the final axis, returning shape `[2, 3, 4]`.
  - Inserting axes with `None`: `x[:, None]` inserts a size-1 dimension at position 1 (equivalent to `.unsqueeze(1)`).
  ```python
  x = torch.zeros(3, 4)
  view_slice = x[:, 1]
  view_slice.fill_(5.0)
  assert x[0, 1] == 5.0  # Original tensor was modified!
  ```
- **Advanced Indexing (Integer tensors or boolean masks):**
  Returns a **COPY**. It allocates new memory. Changing the result does NOT change the original tensor.
  - Coordinate pairs: `x[torch.tensor([0, 2]), torch.tensor([1, 3])]` selects `(0, 1)` and `(2, 3)`.
  - Boolean masks: `x[x % 2 == 0]` selects all even numbers as a flat 1D tensor.
  ```python
  x = torch.zeros(3, 4)
  copy_idx = x[torch.tensor([0, 1])]
  copy_idx.fill_(9.0)
  assert x[0, 0] == 0.0  # Original tensor is untouched!
  ```

---

### 6. `gather`, `scatter_`, and Grouped Reductions

- **`gather` (The Address Book):** "For each row, look at the index to pick which column to read."
  ```python
  # Logits for 2 examples across 3 classes
  scores = torch.tensor([
      [0.1, 0.7, 0.2],  # True class is 1 (score 0.7)
      [0.8, 0.1, 0.1],  # True class is 0 (score 0.8)
  ])
  targets = torch.tensor([[1], [0]])  # Must match dimensions: [2, 1]

  # Read the scores for the true class of each row:
  chosen = scores.gather(dim=1, index=targets)
  assert torch.equal(chosen, torch.tensor([[0.7], [0.8]]))
  ```

- **`scatter_` (Sorting Mail):** "For each row, write the value into the column specified by the index."
  Used to create one-hot encoded vectors from class labels:
  ```python
  import torch.nn.functional as F

  labels = torch.tensor([2, 0, 1])  # Target classes
  one_hot = torch.zeros(3, 3)
  one_hot.scatter_(dim=1, index=labels[:, None], src=1.0)
  # PyTorch provides the built-in helper: F.one_hot(labels, num_classes=3)
  ```

- **`scatter_add_` and `torch.bincount` (Histogram Counting):**
  To count class occurrences or sum values into buckets:
  ```python
  targets = torch.tensor([1, 0, 3, 2, 1, 0], dtype=torch.long)
  # Count frequencies with scatter_add_:
  counts = torch.zeros(4).scatter_add_(dim=0, index=targets, src=torch.ones(6))
  # Or with the built-in helper:
  bincounts = torch.bincount(targets, minlength=4).float()
  assert torch.equal(counts, bincounts)
  ```

---

### 7. Physical Memory: Strides, Storage Offset, and Contiguous Layout

A tensor is a 1D strip of memory in RAM, interpreted as a multi-dimensional grid using **strides** and **storage offset**:
- **Stride:** The number of elements you must skip in physical memory to move one step along an axis.
- **Storage Offset:** The index in the underlying memory buffer where the tensor's first element begins (e.g. slicing `x[1:]` sets offset to 1 row's width).
- **`.is_contiguous()`**: Returns `True` if numbers in memory follow standard row-major reading order without gaps or swaps.

```python
x = torch.arange(24).reshape(4, 6)
# Stride is (6, 1): 6 elements to jump a row, 1 element to jump a column
assert x.stride() == (6, 1)
assert x.is_contiguous() == True
assert x.storage_offset() == 0

# Transposition swaps strides without moving any numbers in RAM:
t = x.t()
assert t.stride() == (1, 6)
assert t.is_contiguous() == False  # Non-contiguous!

# .expand() simulates broadcasting with ZERO stride:
e = x[0:1, :].expand(5, 6)
# Stride for axis 0 is 0 because moving down rows never advances memory!
assert e.stride() == (0, 1)
```

---

### 8. Einstein Summation (`torch.einsum`)

`torch.einsum` allows you to express matrix multiplications, transpositions, dot products, and attention contractions using a concise mathematical string.

**The Rule:**
1. List the indices of the input tensors separated by a comma.
2. Use `->` followed by the indices of the output tensor.
3. Any index present in the inputs but **omitted** from the output is **summed over** (contracted).

```python
# 1. Batched Linear Layer: x [B, T, D] @ W [D, O] -> [B, T, O]
# 'd' appears in both inputs and is omitted in output -> summed over!
x = torch.randn(2, 5, 8)
W = torch.randn(8, 16)
out_einsum = torch.einsum("btd,do->bto", x, W)
out_matmul = x @ W
assert torch.allclose(out_einsum, out_matmul)

# 2. Multi-Head Attention Scores: Q @ K.T -> [B, H, T, S]
# Q: [B, H, T, Dh], K: [B, H, S, Dh] -> Dh is contracted:
q = torch.randn(2, 4, 5, 16)
k = torch.randn(2, 4, 5, 16)
scores_einsum = torch.einsum("bhtd,bhsd->bhts", q, k)
scores_matmul = q @ k.transpose(-2, -1)
assert torch.allclose(scores_einsum, scores_matmul)
```

---

### 9. Sliding Windows (`unfold`), Testing Tools, and Image Patching

#### A. Sliding Windows without Loops: `x.unfold()`
`unfold(dimension, size, step)` extracts consecutive sliding windows from a tensor without copying data:
```python
x = torch.arange(1, 11, dtype=torch.float32)  # [1, 2, ..., 10]
# Extract windows of size 3 with step 1:
windows = x.unfold(dimension=0, size=3, step=1)
# windows shape: [8, 3] -> [[1, 2, 3], [2, 3, 4], ..., [8, 9, 10]]
moving_avg = windows.mean(dim=1)  # Shape [8]
```

#### B. Gold-Standard Unit Testing: `torch.testing.assert_close()`
In modern PyTorch, avoid `assert (a == b).all()`. Instead, use `torch.testing.assert_close(actual, expected)`:
- Checks shape, dtype, and device compatibility.
- Provides rich error messages with relative and absolute mismatch tolerances (`rtol`, `atol`).

#### C. Vision Transformer (ViT) Patch Extraction
A Vision Transformer breaks a 2D image of shape `[B, C, H, W]` into non-overlapping spatial patches of size $P \times P$:
1. Use `reshape` to split $H$ into $(H/P) \times P$ and $W$ into $(W/P) \times P$:
   Shape: `[B, C, H//P, P, W//P, P]`
2. Use `permute` to reorder dimensions into `[B, H//P, W//P, C, P, P]`:
3. Flatten spatial grids into tokens of dimension $P^2 C$:
   Final shape: `[B, (H/P)*(W/P), C*P*P]`
```python
images = torch.randn(4, 3, 32, 32)
P = 8
B, C, H, W = images.shape
# Split into patches:
patches = images.reshape(B, C, H//P, P, W//P, P).permute(0, 2, 4, 1, 3, 5).reshape(B, (H//P)*(W//P), C*P*P)
assert patches.shape == (4, 16, 192)
```

---

### 10. Worked Example: Per-Channel Image Batch Normalization (Reductions & Broadcasting)

In computer vision pipelines, raw input images typically have pixel values in $[0.0, 1.0]$ or $[0, 255]$. Neural networks train fastest and most stably when features have **zero mean** ($\mu = 0$) and **unit variance** ($\sigma = 1$).

Normalizing an image batch across its spatial dimensions requires combining **multi-axis reductions**, **dimension preservation (`keepdim=True`)**, **numerical clamping**, and **broadcasting**.

#### The Problem:
Given a batch of 8 RGB images of size $32 \times 32$:
`images` has shape `[8, 3, 32, 32]` representing `[Batch, Channel, Height, Width]`.
We want to normalize each color channel independently across the entire batch:
$$\text{Normalized}_{b, c, h, w} = \frac{\text{images}_{b, c, h, w} - \mu_c}{\sigma_c + \epsilon}$$

```python
import torch

# 1. Synthesize a batch of 8 RGB images:
torch.manual_seed(42)
images = torch.rand(8, 3, 32, 32, dtype=torch.float32)

# 2. Compute channel mean: squash batch(0), height(2), and width(3)
# keepdim=True preserves the squashed axes as size 1 -> shape [1, 3, 1, 1]
mean = images.mean(dim=(0, 2, 3), keepdim=True)
assert mean.shape == (1, 3, 1, 1)

# 3. Compute channel standard deviation (population std with correction=0)
std = images.std(dim=(0, 2, 3), keepdim=True, correction=0)
assert std.shape == (1, 3, 1, 1)

# 4. Prevent division by zero if an image has constant pixels
std_clamped = std.clamp_min(1e-6)

# 5. Normalize via broadcasting: [8, 3, 32, 32] - [1, 3, 1, 1] -> [8, 3, 32, 32]
# The size 1 at axis 0, 2, and 3 automatically stretches to match 8, 32, and 32!
normalized = (images - mean) / std_clamped
assert normalized.shape == (8, 3, 32, 32)
assert normalized.is_contiguous()

# 6. Scientific verification: check that normalized channel mean is ~0 and std is ~1
norm_mean = normalized.mean(dim=(0, 2, 3))
norm_std = normalized.std(dim=(0, 2, 3), correction=0)

torch.testing.assert_close(norm_mean, torch.zeros(3), atol=1e-5, rtol=1e-5)
torch.testing.assert_close(norm_std, torch.ones(3), atol=1e-5, rtol=1e-5)
print("Image batch normalization verified successfully!")
```

---

### Assignments 2

#### Assignment 2.1 — Shape Prediction and Axis Semantics
- **Objective**: Accurately predict and categorize tensor shape transformations without running code.
- **Given**: Base tensor `x = torch.zeros(2, 3, 4, 5)`.
- **Build / Do**:
  1. Write on paper the predicted output shapes and categorize each transformation (removed, inserted, reordered, merged, or enlarged):
     - `x[0]`
     - `x[0:1]`
     - `x[..., -1]`
     - `x[:, None, :, :, :]`
     - `x.permute(0, 2, 3, 1)`
     - `x.flatten(1, 2)`
     - `torch.stack([x, x], dim=2)`
     - `torch.cat([x, x], dim=1)`
  2. Implement assertion checks for all 8 operations.
- **Pass Criteria**:
  - Assert exact shapes:
    - `x[0].shape == (3, 4, 5)` (axis 0 removed)
    - `x[0:1].shape == (1, 3, 4, 5)` (axis 0 preserved as slice)
    - `x[..., -1].shape == (2, 3, 4)` (last axis removed)
    - `x[:, None, :, :, :].shape == (2, 1, 3, 4, 5)` (axis 1 inserted)
    - `x.permute(0, 2, 3, 1).shape == (2, 4, 5, 3)` (axes reordered)
    - `x.flatten(1, 2).shape == (2, 12, 5)` (axes 1 and 2 merged)
    - `torch.stack([x, x], dim=2).shape == (2, 3, 2, 4, 5)` (new axis inserted at dim 2)
    - `torch.cat([x, x], dim=1).shape == (2, 6, 4, 5)` (axis 1 enlarged from 3 to 6)

#### Assignment 2.2 — Broadcasting Debugger & Loss Bug Analysis
- **Objective**: Master right-to-left broadcasting rules and eliminate silent loss calculation bugs.
- **Given**: Four shape pairs:
  1. `[4, 1, 8]` and `[3, 8]`
  2. `[2, 3]` and `[3, 2]`
  3. `[5, 1]` and `[7]`
  4. `[2, 3, 4]` and `[3, 1]`
- **Build / Do**:
  1. Right-align each pair, check compatibility for each axis, and predict the output shape or `RuntimeError`.
  2. Verify with code:
     - Pair 1: `(4, 3, 8)`
     - Pair 2: Incompatible (raises `RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1`)
     - Pair 3: `(5, 7)`
     - Pair 4: `(2, 3, 4)`
  3. Recreate the common loss bug: predictions `pred = torch.randn(32, 1)` and targets `target = torch.randn(32)`. Show that `(pred - target).shape == (32, 32)`, and write the fix using `unsqueeze(1)`.
- **Pass Criteria**:
  - All 4 predictions validated by executable code.
  - Written explanation of why `(pred - target)` produces a $32 \times 32$ matrix instead of a $32 \times 1$ vector and how this corrupts MSE or L1 loss.

#### Assignment 2.3 — Indexing Laboratory: Views vs. Copies
- **Objective**: Distinguish basic indexing (views) from advanced indexing (copies) and master multidimensional selections.
- **Given**: `x = torch.arange(20).reshape(4, 5)`.
- **Build / Do**:
  1. Select rows `3, 1, 1` using integer tensor indexing: `x[torch.tensor([3, 1, 1])]`.
  2. Select columns `4, 0, 2`: `x[:, torch.tensor([4, 0, 2])]`.
  3. Select coordinate pairs $(0, 4)$, $(2, 0)$, and $(3, 2)$: `x[torch.tensor([0, 2, 3]), torch.tensor([4, 0, 2])]`.
  4. Select the Cartesian submatrix of rows `[0, 3]` and columns `[1, 2, 4]` using broadcasted index vectors.
  5. Select all elements divisible by 3 using a boolean mask: `x[x % 3 == 0]`.
  6. Prove that basic slice `x[0, :]` shares memory (mutating it changes `x`), while advanced indexing `x[torch.tensor([0])]` produces a copy.
- **Pass Criteria**:
  - Assert exact shapes and values for all 5 selections:
    - Rows `[3, 1, 1]`: shape `(3, 5)`
    - Columns `[4, 0, 2]`: shape `(4, 3)`
    - Coordinate pairs: shape `(3,)`, values `[4, 10, 17]`
    - Cartesian block: shape `(2, 3)`, values `[[1, 2, 4], [16, 17, 19]]`
    - Divisible by 3: shape `(7,)`, values `[0, 3, 6, 9, 12, 15, 18]`
  - Explicit assertion confirming mutation of basic slice affects `x` and mutation of advanced index does not.

#### Assignment 2.4 — Gather, Scatter, and Grouped Reductions
- **Objective**: Implement indexed extraction, one-hot encoding, and histogram counting using `gather` and `scatter`.
- **Given**:
  - Logits matrix: `logits = torch.randn(6, 4)`
  - Class targets: `targets = torch.tensor([1, 0, 3, 2, 1, 0], dtype=torch.long)`
- **Build / Do**:
  1. Use `gather(dim=1, index=targets[:, None])` to extract the predicted logit corresponding to the true target for each sample.
  2. Use `torch.zeros(6, 4).scatter_(dim=1, index=targets[:, None], src=1.0)` to generate a one-hot label matrix.
  3. Use `torch.zeros(4).scatter_add_(dim=0, index=targets, src=torch.ones(6))` to count the frequency of each class.
- **Pass Criteria**:
  - Assert that `gathered[:, 0]` matches manual loop extraction `torch.tensor([logits[i, targets[i]] for i in range(6)])`.
  - Assert that one-hot matrix matches `F.one_hot(targets, num_classes=4).float()`.
  - Assert that class counts match `torch.bincount(targets, minlength=4).float()`.

#### Assignment 2.5 — Vectorized Sliding-Window Moving Average
- **Objective**: Compute moving averages efficiently using `unfold` without Python loops.
- **Given**: 1D sequence `x = torch.arange(1, 11, dtype=torch.float32)` (values 1 through 10), window size $W = 3$.
- **Build / Do**:
  1. Use `x.unfold(dimension=0, size=3, step=1)` to generate consecutive sliding windows.
  2. Compute the mean along the unfolded window dimension (`dim=1`).
- **Pass Criteria**:
  - No Python loops or list comprehensions used.
  - Assert output shape is `(8,)`.
  - Assert exact values: `torch.testing.assert_close(result, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))`.

#### Assignment 2.6 — Einstein Summation vs. Standard Operations
- **Objective**: Translate tensor contractions between mathematical notation, `torch.einsum`, and standard batched matrix multiplication (`@`).
- **Given**:
  - Feature tensor `x` of shape `[B, T, D] = [2, 5, 8]`
  - Linear weight `W` of shape `[D, O] = [8, 16]` and bias `b` of shape `[O] = [16]`
  - Query/Key tensors `q, k` of shape `[B, H, T, Dh] = [2, 4, 5, 16]`
- **Build / Do**:
  1. Compute linear projection with `einsum`: `torch.einsum("btd,do->bto", x, W) + b`.
  2. Compute linear projection with `@`: `(x @ W) + b`.
  3. Compute unscaled attention scores with `einsum`: `torch.einsum("bhtd,bhsd->bhts", q, k)`.
  4. Compute attention scores with `@`: `q @ k.transpose(-2, -1)`.
- **Pass Criteria**:
  - Assert `torch.allclose` between `einsum` and `@` linear outputs.
  - Assert `torch.allclose` between `einsum` and `@` attention scores of shape `[2, 4, 5, 5]`.
  - Provide written explanation identifying which indices are contracted (reduced) in each equation (`d` in linear, `Dh` in attention).

#### Assignment 2.7 — Storage, Strides, and Contiguity Investigation
- **Objective**: Inspect the physical memory representation of tensor views, transpositions, and copies.
- **Given**: `x = torch.arange(24).reshape(4, 6)`.
- **Build / Do**:
  1. Create:
     - Slice: `s = x[:2, :3]`
     - Transpose: `t = x.t()`
     - Advanced index: `a = x[[0, 1], :]`
     - Cloned copy: `c = x.clone()`
     - Detached view: `d = x.detach()`
     - Expanded view: `e = x[0:1, :].expand(5, 6)`
  2. For each tensor, record: `shape`, `stride()`, `storage_offset()`, `is_contiguous()`, and whether in-place mutation alters `x`.
- **Pass Criteria**:
  - Produce a structured table recording these 5 properties for all cases.
  - Verify that `t.stride() == (1, 6)` and `t.is_contiguous() == False`.
  - Verify that `e.stride() == (0, 1)` (zero stride on expanded dimension).
  - Verify that mutating `s` or `d` alters `x`, while mutating `a` or `c` does not.

#### Assignment 2.8 — Patch Extraction: Native Operations vs. Einops
- **Objective**: Implement Vision Transformer (ViT) patch extraction and spatial reconstruction using native PyTorch operations.
- **Given**: Image batch `images = torch.randn(4, 3, 32, 32)` and patch size $P = 8$ (grid size $H/P = 4, W/P = 4$, total patches = 16, patch dimension = $3 \times 8 \times 8 = 192$).
- **Build / Do**:
  1. Reshape and permute `images` to patch representation of shape `[4, 16, 192]` using only native `reshape` and `permute`.
  2. Invert the transformation to reconstruct the original `[4, 3, 32, 32]` tensor from patches.
  3. (Optional/Reference): Write the equivalent `einops` pattern string (`"b c (h p1) (w p2) -> b (h w) (c p1 p2)"`).
- **Pass Criteria**:
  - Assert patch tensor shape is exactly `(4, 16, 192)`.
  - Assert reconstructed tensor matches input: `torch.testing.assert_close(reconstructed, images)`.

#### Assignment 2.9 — Per-Channel Image Normalization Mechanics
- **Objective**: Implement per-channel batch normalization from scratch using multi-axis reductions, dimension preservation, and broadcasting.
- **Given**:
  - Random image batch: `images = torch.rand(16, 3, 64, 64, dtype=torch.float32)`
  - Degenerate constant batch: `constant_images = torch.full((4, 3, 16, 16), 5.0, dtype=torch.float32)`
- **Build / Do**:
  1. Compute per-channel mean and standard deviation (`correction=0`) across batch, height, and width (`dim=(0, 2, 3)`) preserving dimensions with `keepdim=True`.
  2. Guard against division by zero using `clamp_min(1e-6)`.
  3. Normalize the batch via broadcasted subtraction and division.
  4. Repeat normalization on `constant_images` and demonstrate that clamping prevents `NaN` and `Inf`.
  5. Scientifically verify that the normalized output for `images` has per-channel mean $\approx 0.0$ and std $\approx 1.0$ using `torch.testing.assert_close`.
- **Pass Criteria**:
  - Assert output shape is strictly `(16, 3, 64, 64)`.
  - Assert `torch.testing.assert_close(norm_mean, torch.zeros(3), atol=1e-5, rtol=1e-5)`.
  - Assert `torch.testing.assert_close(norm_std, torch.ones(3), atol=1e-5, rtol=1e-5)`.
  - Assert `torch.isfinite(norm_constant).all()` evaluates to `True`.

**Exit criterion:** You can predict shapes, choose basic versus advanced indexing, use gather/scatter, express contractions, and identify when an operation is a view or copy.

---

## Chapter 3 — Autograd: reverse-mode differentiation

### What are we learning in this chapter?
- **The Autograd Mental Model**: How PyTorch logs operations in a Directed Acyclic Graph (DAG) during the forward pass and walks backward using the chain rule.
- **Leaves vs. Intermediate Tensors**: The difference between user-created parameters (`is_leaf=True`) and intermediate activations.
- **The `grad_fn` Attribute**: How PyTorch attaches backward operation objects to every computed tensor.
- **Why Gradients Accumulate**: The critical reason why PyTorch *adds* to `.grad` instead of overwriting it, and why `optimizer.zero_grad(set_to_none=True)` is mandatory.
- **Controlling Gradient Tracking**: When and why to use `torch.no_grad()`, `torch.inference_mode()`, and `tensor.detach()`.
- **Numerical Gradient Verification**: Implementing finite differences to scientifically verify that analytical gradients are correct.
- **Debugging Broken Gradient Paths**: Identifying silent bugs where gradients fail to flow back to parameters.

---

### 1. What problem does Autograd solve?

In Chapter 0, we calculated derivatives by hand for two simple parameters: $w$ and $b$.
Now imagine a modern neural network:
- It has 100 layers.
- It contains 50,000,000 parameters.
- It uses convolutions, attention, residual connections, and normalizations.

Deriving the calculus formulas by hand on paper would take months and would be riddled with human error.
**Autograd (Automatic Differentiation)** completely solves this:
> **As you run your PyTorch Python code forward, Autograd records every mathematical operation into a computation graph. When you call `.backward()`, Autograd automatically applies the chain rule in reverse to compute exact derivatives for all your parameters.**

---

### 2. The Mental Model: The Recipe Notebook

Think of Autograd as a chef taking notes while cooking:
1. **Forward Pass:** Every time you add, multiply, or pass a tensor through a function, PyTorch writes that step down in a notebook (the **computation graph**).
   - If a tensor was created by you (like a model weight) with `requires_grad=True`, it is called a **leaf tensor**.
   - If a tensor was created by math (like `pred = w * x + b`), it is a **non-leaf tensor**. PyTorch attaches a `grad_fn` (gradient function) to it, remembering: "This tensor was created by an addition operation."
2. **Backward Pass (`loss.backward()`):**
   - PyTorch starts at the final scalar loss at the end of the notebook.
   - It reads the notebook **backwards**, applying the calculus chain rule step-by-step.
   - For every leaf parameter, it calculates the gradient and stores it in `parameter.grad`.

```
FORWARD PASS: (Building the Graph)
x [leaf] ----\
              * [MulBackward0] --> a ----\
w [leaf] ----/                            + [AddBackward0] --> loss
                                         /
b [leaf] -------------------------------/

BACKWARD PASS: (Walking Backwards via Chain Rule)
x.grad <----\
              * <----------------- a.grad <----\
w.grad <----/                                   + <---- loss.backward()
                                               /
b.grad <--------------------------------------/
```

---

### 3. Step-by-Step with Real Numbers: The Chain Rule

Let's see the chain rule in action with simple math:
$$x = 2.0, \quad a = 3x, \quad b = \sin(a), \quad \text{loss} = b^2$$

Let's trace backwards from `loss`:
1. How does `loss` change when $b$ changes?
   $$\frac{\partial \text{loss}}{\partial b} = 2b$$
2. How does $b$ change when $a$ changes?
   $$\frac{\partial b}{\partial a} = \cos(a)$$
3. How does $a$ change when $x$ changes?
   $$\frac{\partial a}{\partial x} = 3$$

To find how `loss` changes when $x$ changes, **multiply the chain of sensitivities together**:
$$\frac{d\text{loss}}{dx} = \frac{\partial \text{loss}}{\partial b} \times \frac{\partial b}{\partial a} \times \frac{\partial a}{\partial x} = (2b) \times (\cos(a)) \times (3)$$

Let's check with PyTorch:
```python
import torch, math

x = torch.tensor(2.0, requires_grad=True)
a = 3 * x
b = torch.sin(a)
loss = b ** 2

loss.backward()

# Analytical answer: 2 * sin(6) * cos(6) * 3 = 3 * sin(12)
expected = 3.0 * math.sin(12.0)
assert torch.isclose(x.grad, torch.tensor(expected))
print(f"Autograd calculated: {x.grad.item():.4f}, exact analytical: {expected:.4f}")
```

---

### 4. Why Gradients Accumulate (And The `zero_grad` Requirement)

In PyTorch, calling `loss.backward()` **adds** the newly computed gradients to whatever is already stored in `.grad`:

$$\text{parameter.grad} = \text{parameter.grad} + \text{new\_gradient}$$

**Why did PyTorch designers do this?**
Because of **Gradient Accumulation**!
If your GPU is too small to fit a batch of 64 images, you can run 4 batches of 16 images one after another, calling `.backward()` on each. Because gradients accumulate, after 4 batches the parameter `.grad` contains the exact sum of gradients for all 64 images!

**The Catch:**
If you don't reset the gradients before the next training step, the gradients will keep growing bigger and bigger every epoch until your loss explodes!
```python
# The standard way:
optimizer.zero_grad(set_to_none=True)

# Or manually:
w.grad = None  # Setting to None frees memory immediately!
```

---

### 5. Turning Off Autograd: `no_grad` vs. `inference_mode`

When evaluating a model or running inference, you don't need gradients.
Tracking the computation graph uses memory and CPU/GPU time.

PyTorch provides two context managers:
1. `with torch.no_grad():`
   - Disables gradient computation.
   - Tensors can still be used in autograd later if needed.
2. `with torch.inference_mode():` (Preferred in PyTorch 2!)
   - Disables gradient computation *and* view tracking/version counters.
   - Faster and uses less memory than `no_grad`.

---

### 6. Finite Differences: Verifying Autograd Scientifically

How do you know Autograd isn't lying to you?
You can verify analytical and autograd gradients using the **central finite difference** formula:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

- $\epsilon$ is an extremely small step (e.g. $10^{-4}$).
- $e_i$ is a unit vector pointing along axis $i$.

> [!IMPORTANT]
> **Use `torch.float64` (Double Precision):**
> When subtracting two nearly identical numbers ($f(x + \epsilon) - f(x - \epsilon)$), standard 32-bit floats suffer from **catastrophic numerical cancellation**. Always perform gradient checking in `torch.float64`!

```python
import torch

def f(x):
    return x[0]**3 + x[0]*x[1] + torch.sin(x[1])

x = torch.tensor([2.0, -0.785398], dtype=torch.float64, requires_grad=True)
loss = f(x)
loss.backward()
autograd_grad = x.grad.clone()

# Numerical approximation:
eps = 1e-4
num_grad = torch.zeros_like(x)
for i in range(len(x)):
    x_pos = x.detach().clone()
    x_neg = x.detach().clone()
    x_pos[i] += eps
    x_neg[i] -= eps
    num_grad[i] = (f(x_pos) - f(x_neg)) / (2 * eps)

assert torch.allclose(autograd_grad, num_grad, rtol=1e-4, atol=1e-5)
```

---

### 7. Non-Linear Activations: Why XOR Needs Curvature

A single linear equation $\hat{y} = X w + b$ can only draw a straight line.
In the classic **XOR (Exclusive OR)** problem:
- $(0, 0) \to 0$
- $(1, 1) \to 0$
- $(0, 1) \to 1$
- $(1, 0) \to 1$

No straight line in the world can separate $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$!
To solve this, we must pass linear combinations through a **non-linear activation function**:
- **`torch.tanh(x)`**: S-shaped curve squashing inputs to $(-1, 1)$.
- **`torch.sigmoid(x)`**: S-shaped curve squashing inputs to $(0, 1)$.

A simple 2-layer network with non-linearity bends the input space so XOR becomes linearly separable:
```python
# Hidden layer with non-linearity:
hidden = torch.tanh(X @ W1 + b1)
# Output prediction:
pred = torch.sigmoid(hidden @ W2 + b2)
```

---

### 8. Diagnosing the 4 Silent Gradient Killers

When building neural networks, gradients can silently stop flowing. Here are the 4 most common traps:

1. **Premature Detach (`.detach()`):**
   ```python
   hidden = (X @ W1).detach()  # Severed the computation graph!
   loss = (hidden @ W2).sum()
   loss.backward()  # W1.grad will be None!
   ```
   *Fix:* Only detach tensors when you deliberately want to stop gradients.

2. **In-Place Mutation of Activations Needed for Backward:**
   ```python
   a = torch.relu(x)
   a.zero_()  # IN-PLACE MUTATION! Overwrote the values Autograd needs for the chain rule!
   ```
   *Result:* PyTorch throws `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
   *Fix:* Avoid in-place methods (`zero_()`, `add_()`, `fill_()`) on tensors that participate in autograd.

3. **Leaving the PyTorch Graph via NumPy:**
   ```python
   h = torch.from_numpy(hidden.numpy())  # PyTorch has no idea what happened in NumPy!
   ```
   *Fix:* Perform all math using native `torch` operations.

4. **Re-wrapping Tensors:**
   ```python
   y_pred = torch.tensor(hidden.tolist())  # Creates a brand-new leaf tensor, cutting off history!
   ```
   *Fix:* Pass tensors directly through operations without converting to Python lists.

---

### 9. Higher-Order Derivatives: `create_graph=True`

Normally, calling `loss.backward()` destroys the intermediate computation graph to save memory.
What if you want the **second derivative** (curvature, Hessian, or physics-informed neural networks)?
You must tell Autograd to **retain and build a graph of the backward pass itself**:

```python
x = torch.tensor(2.0, requires_grad=True)
# f(x) = x^4 - 3x^3 + 2x^2
y = x**4 - 3*x**3 + 2*x**2

# First derivative: f'(x) = 4x^3 - 9x^2 + 4x -> at x=2: 32 - 36 + 8 = 4
# create_graph=True tracks the differentiation steps so we can differentiate again!
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
assert torch.isclose(grad1, torch.tensor(4.0))

# Second derivative: f''(x) = 12x^2 - 18x + 4 -> at x=2: 48 - 36 + 4 = 16
grad2 = torch.autograd.grad(grad1, x)[0]
assert torch.isclose(grad2, torch.tensor(16.0))
```

---

### Assignments 3

#### Assignment 3.1 — Follow the Chain by Hand
- **Objective**: Verify that autograd's reverse-mode differentiation matches analytical calculus step by step.
- **Given**:
  - Input: $x = 2.0$
  - Intermediate nodes: $a = 3x$, $b = \sin(a)$, $\text{loss} = b^2$.
- **Build / Do**:
  1. Manually calculate local derivatives:
     - $\frac{\partial \text{loss}}{\partial b} = 2b$
     - $\frac{\partial b}{\partial a} = \cos(a)$
     - $\frac{\partial a}{\partial x} = 3$
  2. Compute analytical chain rule: $\frac{d\text{loss}}{dx} = 2\sin(3x) \cdot \cos(3x) \cdot 3 = 6\sin(6)\cos(6) = 3\sin(12)$.
  3. Implement with PyTorch:
     ```python
     x = torch.tensor(2.0, requires_grad=True)
     a = 3 * x
     b = torch.sin(a)
     loss = b ** 2
     loss.backward()
     ```
- **Pass Criteria**:
  - Assert that `loss.grad_fn` is `PowBackward0` and `b.grad_fn` is `SinBackward0`.
  - Assert `torch.isclose(x.grad, torch.tensor(3.0 * math.sin(12.0)))`.

#### Assignment 3.2 — Vector Linear Regression Gradients via Autograd
- **Objective**: Validate autograd gradients against analytical matrix calculus for multivariate regression.
- **Given**:
  - Input: `X = torch.randn(100, 3)`
  - True weights: `w_true = torch.tensor([1.5, -2.0, 3.0])`
  - Targets: `y = X @ w_true + 0.5`
  - Initial weights: `w = torch.zeros(3, requires_grad=True)`, `b = torch.zeros((), requires_grad=True)`
- **Build / Do**:
  1. Compute prediction: `pred = X @ w + b`.
  2. Compute loss: `loss = ((pred - y) ** 2).mean()`.
  3. Run `loss.backward()`.
  4. Compute analytical gradients:
     - `w_grad_analytical = (2.0 / 100) * (X.t() @ (pred.detach() - y))`
     - `b_grad_analytical = (2.0 / 100) * (pred.detach() - y).sum()`
- **Pass Criteria**:
  - Assert `torch.testing.assert_close(w.grad, w_grad_analytical)`.
  - Assert `torch.testing.assert_close(b.grad, b_grad_analytical)`.

#### Assignment 3.3 — Finite-Difference Gradient Checker
- **Objective**: Implement a numerical gradient approximation to check analytical/autograd gradients.
- **Given**: Scalar function $f(x) = x_0^3 + x_0 x_1 + \sin(x_1)$ evaluated at $x = [2.0, -\frac{\pi}{4}]$.
- **Build / Do**:
  1. Compute exact gradients using autograd.
  2. Implement central finite difference approximation for each component $i$:
     $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
     with $\epsilon = 10^{-4}$ in `float64`.
- **Pass Criteria**:
  - Perform all computations in `torch.float64` to prevent numerical cancellation.
  - Assert `torch.allclose(autograd_grad, numerical_grad, rtol=1e-4, atol=1e-5)`.

#### Assignment 3.4 — XOR Solver Without `nn.Module`
- **Objective**: Train a multi-layer perceptron to solve the non-linear XOR problem using only raw tensors and autograd.
- **Given**:
  - XOR inputs: `X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])`
  - Targets: `y = torch.tensor([[0.], [1.], [1.], [0.]])`
  - Hidden layer dimension $H = 4$.
- **Build / Do**:
  1. Initialize weights and biases with `requires_grad=True`: `W1` ($2 \times 4$), `b1` ($4$), `W2` ($4 \times 1$), `b2` ($1$).
  2. Write training loop: forward pass with `torch.tanh` or `torch.sigmoid`, binary cross-entropy or MSE loss, `loss.backward()`.
  3. Manually update parameters inside `with torch.no_grad():` and zero gradients with `.grad = None`.
- **Pass Criteria**:
  - Model trains until MSE $< 0.01$ or binary accuracy is $100\%$ on all four inputs.
  - Predictions rounded to nearest integer match `y` exactly.

#### Assignment 3.5 — Diagnosing Four Broken Gradient Paths
- **Objective**: Identify, diagnose, and fix common autograd bugs that silently break gradient propagation.
- **Given**: Four broken forward pass scenarios:
  1. Tensor detached prematurely: `hidden = (X @ W1).detach()`
  2. In-place modification of activation needed for backward: `a = torch.relu(x); a.zero_()`
  3. Conversion through NumPy: `h = torch.from_numpy(hidden.numpy())`
  4. Creating a new tensor without tracking: `y_pred = torch.tensor(hidden.tolist())`
- **Build / Do**:
  1. Execute backward on each broken scenario and capture the symptom (e.g. `w.grad is None`, `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`).
  2. Implement the minimal fix for each case to restore the unbroken gradient path.
- **Pass Criteria**:
  - For all four cases, record the exact error/symptom and verify that the fixed version produces valid, non-zero `.grad` values.

#### Assignment 3.6 — Second-Order Derivatives via `create_graph`
- **Objective**: Compute second-order derivatives (Hessian vector product or curvature) using autograd graph retention.
- **Given**: Scalar function $f(x) = x^4 - 3x^3 + 2x^2$ evaluated at $x = 2.0$.
- **Build / Do**:
  1. Compute first derivative with `torch.autograd.grad(f(x), x, create_graph=True)[0]`.
  2. Compute second derivative by differentiating the first derivative with respect to $x$.
  3. Compare with analytical derivatives:
     - $f'(x) = 4x^3 - 9x^2 + 4x \implies f'(2) = 32 - 36 + 8 = 4$.
     - $f''(x) = 12x^2 - 18x + 4 \implies f''(2) = 48 - 36 + 4 = 16$.
- **Pass Criteria**:
  - Assert `torch.isclose(grad1, torch.tensor(4.0))`.
  - Assert `torch.isclose(grad2, torch.tensor(16.0))`.

#### Assignment 3.7 — Autograd Closed-Book Retrieval
- **Objective**: Demonstrate complete fluency with autograd semantics from scratch.
- **Given**: An empty file.
- **Build / Do**:
  - Implement a 2-layer polynomial regression with custom loss, backpropagation, and manual gradient step without consulting notes.
- **Pass Criteria**:
  - Program runs without error, verifies `.grad` shapes, and decreases loss over 50 steps.

**Exit criterion:** You can explain how autograd constructs the graph, why gradients accumulate, and how to verify gradients with finite differences.

---

# Part II — Training Systems

## Chapter 4 — `nn.Module`: parameters, state, and composition

### What are we learning in this chapter?
- **Why `nn.Module` Exists**: Why functions are not enough (models need *state*—weights and biases that survive across batches).
- **Parameters vs. Buffers**: The vital difference between learnable weights (`nn.Parameter`) and persistent non-gradient state (`register_buffer`).
- **Child Module Containers**: Why assigning layers to a plain Python list fails to register parameters, and how `nn.ModuleList` and `nn.Sequential` fix it.
- **Built-in Layer Geometries**: Understanding shape contracts for `Linear`, `Embedding`, `Conv2d`, `BatchNorm2d`, and `LayerNorm`.
- **Loss Functions & Numerical Stability**: Why passing `softmax(logits)` to `CrossEntropyLoss` causes catastrophic underflow, and how `BCEWithLogitsLoss` solves numerical instability.
- **Module Modes**: What `model.train()` and `model.eval()` actually change under the hood (BatchNorm and Dropout behavior).

---

### 1. What problem does `nn.Module` solve?

In pure Python or mathematical code, a function is **stateless**:
```python
def linear(x, w, b):
    return x @ w.t() + b
```
Every time you call `linear()`, you have to pass $w$ and $b$ in from the outside.
In a deep neural network with 50 layers:
- You would have to manually keep track of 100 separate weight and bias tensors.
- You would have to manually move all 100 tensors to the GPU: `w1 = w1.to('cuda')`, `b1 = b1.to('cuda')`...
- You would have to manually pass all 100 tensors to the optimizer.
- You would have to manually package all 100 tensors to save a checkpoint.

`nn.Module` is PyTorch's elegant solution:
> **An `nn.Module` is a stateful object that bundles together parameters, persistent buffers, child layers, and the forward computation into a single self-contained unit.**

When you call `model.to("cuda")`, PyTorch automatically finds every parameter inside the model and moves it to the GPU.
When you call `model.parameters()`, PyTorch automatically hands all learnable tensors to your optimizer.

---

### 2. Parameters vs. Buffers: The Two Types of State

A neural network module can hold two types of internal state:

1. **Parameters (`nn.Parameter`):**
   - Tensors that get updated by gradient descent (e.g. weights and biases).
   - Created with: `self.weight = nn.Parameter(torch.randn(out_features, in_features))`
   - Automatically included in `model.parameters()` and updated by the optimizer.
   - **`model.parameters()` vs. `model.named_parameters()`:**
     - `model.parameters()` yields each parameter tensor directly.
     - `model.named_parameters()` yields `(name, parameter)` pairs, giving the full hierarchical string path of each weight (e.g. `"layers.0.weight"`), which is essential for inspecting freeze status or configuring parameter groups.

2. **Persistent Buffers (`self.register_buffer`):**
   - Tensors that need to be saved in checkpoints and moved to the GPU with `.to(device)`, but are **NOT** trained by gradients (e.g. running mean/variance in BatchNorm, or a running class counter).
   - Created with: `self.register_buffer("running_mean", torch.zeros(num_features))`
   - Excluded from `model.parameters()`, but included in `model.state_dict()`.
   - Access all buffers via `model.buffers()` or `(name, buffer)` pairs via `model.named_buffers()`.

---

### 3. The Python List Trap: Why `nn.ModuleList` is Required

Suppose you want to create a model with 3 linear layers:
```python
class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        # WRONG! A plain Python list hides layers from PyTorch!
        self.layers = [nn.Linear(4, 4) for _ in range(3)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```
If you run `list(BrokenModel().parameters())`, the list is **EMPTY**!
Why? Because Python lists are not PyTorch modules. PyTorch cannot inspect their contents to register parameters. If you call `model.to('cuda')`, the layers remain stuck on the CPU!

**The Fix:** Use `nn.ModuleList` or `nn.Sequential`:
```python
class WorkingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CORRECT: nn.ModuleList registers all parameters properly!
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
```

---

### 4. Loss Functions: Why Logits Rule the World

One of the most dangerous rookie mistakes is applying `softmax` before `CrossEntropyLoss`:
```python
# CATASTROPHIC BUG!
probs = F.softmax(logits, dim=-1)
loss = nn.CrossEntropyLoss()(probs, targets)  # WRONG!
```

**Why is this wrong?**
1. `nn.CrossEntropyLoss` **already computes log-softmax internally**! If you apply softmax first, you are computing softmax twice, corrupting your loss and gradients.
2. **Numerical Underflow:** If a prediction is very confident (say, a logit of $-100$), `softmax` rounds the probability to pure `0.0`. Taking the natural logarithm with `torch.log(0.0)` produces `-inf`, destroying your network with `NaN`!
`CrossEntropyLoss` uses the mathematical identity $\log(\sum e^x) = c + \log(\sum e^{x-c})$ (the **log-sum-exp trick**) to guarantee numerical stability even for extreme values.

**Module vs. Functional Interface:**
PyTorch provides two equivalent ways to compute losses:
- **Class/Module form (`nn.CrossEntropyLoss`):** A stateful module class initialized once: `criterion = nn.CrossEntropyLoss()`.
- **Functional form (`F.cross_entropy`):** A stateless function `F.cross_entropy(logits, targets)` from `torch.nn.functional` that computes the loss directly without instantiating an object. Both produce identical results.

> [!IMPORTANT]
> **Golden Rule:** Neural network classification heads must ALWAYS output raw, unnormalized **logits**. Let the loss function handle the probabilities.

---

### 5. Module Modes: `train()` vs. `eval()`

Calling `model.train()` or `model.eval()` does **not** freeze weights or stop gradients.
It only flips a switch for layers that behave differently during training vs. testing:
- **`nn.Dropout`:** During `train()`, randomly zeros out neurons. During `eval()`, turns off completely (acts as an identity pass-through).
- **`nn.BatchNorm2d`:** During `train()`, computes mean and variance from the current minibatch and updates running stats. During `eval()`, freezes running stats and uses them to normalize incoming samples.

---

### 6. Building Custom Layers & Parameter Initialization (`nn.init`)

To build your own layer from scratch, subclass `nn.Module`:
1. Define weights and biases inside `__init__` using `nn.Parameter`.
2. Initialize them with **`torch.nn.init`** inside a dedicated `reset_parameters()` method.
3. Define the computation inside `forward(self, x)`.

```python
import math
import torch
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Allocate uninitialized memory for parameters:
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform initialization (He init for LeakyReLU/ReLU):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Bound calculation matching official PyTorch Linear:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        # x is [B, in_features], weight is [out_features, in_features]
        # x @ weight.t() produces [B, out_features]
        return x @ self.weight.t() + (self.bias if self.bias is not None else 0)
```

---

### 7. The Core PyTorch Layer Zoo & Shape Contracts

Every built-in layer has an exact geometric shape contract:

| Layer | Expected Input Shape | Output Shape | What it does |
| :--- | :--- | :--- | :--- |
| **`nn.Linear(10, 5)`** | `[B, 10]` | `[B, 5]` | Matrix multiplication: $x W^T + b$ |
| **`nn.Embedding(100, 32)`** | `[B, T]` (int64 IDs) | `[B, T, 32]` | Table lookup converting token IDs to vectors |
| **`nn.Conv1d(16, 32, 3, padding=1)`** | `[B, 16, L]` | `[B, 32, L]` | 1D temporal/signal convolution |
| **`nn.Conv2d(3, 64, 3, padding=1)`** | `[B, 3, H, W]` | `[B, 64, H, W]` | 2D image convolution |
| **`nn.MaxPool2d(2, 2)`** | `[B, C, H, W]` | `[B, C, H/2, W/2]` | Spatial downsampling (keeps maximum) |
| **`nn.AdaptiveAvgPool2d((1, 1))`**| `[B, C, H, W]` | `[B, C, 1, 1]` | Downsamples spatial grid to exact target size $(1, 1)$ |
| **`nn.GRU(16, 32, batch_first=True)`**| `[B, T, 16]` | `[B, T, 32]` | Recurrent sequence processing |
| **`nn.BatchNorm2d(64)`** | `[B, 64, H, W]` | `[B, 64, H, W]` | Channel-wise normalization across batch and space |
| **`nn.LayerNorm(32)`** | `[B, T, 32]` | `[B, T, 32]` | Feature normalization across the last dimension |

---

### 8. Loss Function Contracts: Shapes, Types, and Targets

Never guess your loss function arguments! Each has a strict contract:

1. **`nn.MSELoss()` (Regression):**
   - Predictions: `[B, 1]` or `[B, D]` (float32).
   - Targets: **Must match shape exactly!** `[B, 1]` or `[B, D]` (float32).
   - *Danger:* Passing `pred` of `[B, 1]` and `target` of `[B]` triggers silent broadcasting to `[B, B]`!

2. **`nn.CrossEntropyLoss()` (Multi-Class Classification):**
   - Predictions: Raw logits of shape `[B, Num_Classes]` (float32).
   - Targets: 1D class indices of shape `[B]` with **`torch.int64` (long)** dtype!
   - Values must be integers: $0 \le \text{target}_i < \text{Num\_Classes}$.

3. **`nn.BCEWithLogitsLoss()` (Binary / Multi-Label Classification):**
   - Predictions: Raw unnormalized logits of shape `[B, 1]` or `[B, Num_Labels]` (float32).
   - Targets: **`torch.float32`** tensor of shape `[B, 1]` or `[B, Num_Labels]` with values $0.0$ or $1.0$.

4. **Numerical Stability with `F.log_softmax`:**
   Instead of computing $\log(\text{softmax}(x))$, PyTorch uses the **log-sum-exp trick**:
   $$\log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = x_i - \log\sum_j e^{x_j} = x_i - c - \log\sum_j e^{x_j - c}$$
   where $c = \max(x)$. This eliminates exponential overflow and underflow completely.

---

### 9. State Dictionaries & Checkpointing: `state_dict()`

A **`state_dict`** is a standard Python dictionary mapping the name of every parameter and buffer to its underlying tensor:
- `model.state_dict()`: Returns all weights, biases, and registered buffers.
- `model.load_state_dict(state_dict)`: Copies parameter values from the dictionary into the model.

```python
# 1. Saving a model checkpoint to disk:
torch.save(model.state_dict(), "model.pt")

# 2. Loading a model checkpoint from disk:
model = MyLinear(8, 4)
model.load_state_dict(torch.load("model.pt"))
```

> [!NOTE]
> Buffers registered via `self.register_buffer("name", tensor)` are included in `model.state_dict()` so they are saved to disk, but are excluded from `model.parameters()` so your optimizer will not update them with gradients.

---

### Assignments 4

#### Assignment 4.1 — Rebuild `Linear` from First Principles
- **Objective**: Implement a custom linear layer subclassing `nn.Module` to understand parameter registration and initialization.
- **Given**: Input dimension $D_{in} = 8$, output dimension $D_{out} = 4$.
- **Build / Do**:
  1. Define class `MyLinear(nn.Module)`:
     - Initialize `self.weight = nn.Parameter(torch.empty(out_features, in_features))`.
     - Initialize `self.bias = nn.Parameter(torch.empty(out_features))` if `bias=True`.
     - Implement Kaiming uniform initialization in `reset_parameters()`.
     - Implement `forward(self, x)`: `x @ self.weight.t() + self.bias`.
  2. Instantiate both `MyLinear(8, 4)` and official `nn.Linear(8, 4)`.
  3. Copy weights and biases from `nn.Linear` to `MyLinear`.
- **Pass Criteria**:
  - Assert that `list(my_layer.parameters())` returns both weight and bias tensors.
  - Given identical input `x = torch.randn(5, 8)`, assert `torch.testing.assert_close(my_layer(x), official_layer(x))`.

#### Assignment 4.2 — Submodule & Parameter Registration Failure Modes
- **Objective**: Understand why assigning submodules or parameters to plain Python lists fails to register them.
- **Given**: Three container models:
  - `ModelBad`: stores 3 linear layers in a plain Python list: `self.layers = [nn.Linear(4, 4) for _ in range(3)]`.
  - `ModelList`: stores layers in `self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])`.
  - `ModelSeq`: stores layers in `self.layers = nn.Sequential(...)`.
- **Build / Do**:
  1. Inspect `len(list(m.parameters()))` for all three models.
  2. Call `m.to("cuda")` (or another device) on each model and inspect `m.layers[0].weight.device`.
- **Pass Criteria**:
  - Prove that `ModelBad.parameters()` is empty (`len == 0`) and `.to(device)` fails to move child layers.
  - Prove that `ModelList` and `ModelSeq` register all parameters and move correctly with `.to(device)`.

#### Assignment 4.3 — Built-in Layer Shape Tour
- **Objective**: Master the expected input and output tensor shapes for standard PyTorch layers.
- **Given**: One valid synthetic input for each layer type.
- **Build / Do**:
  - Predict and verify input/output shapes for:
    1. `nn.Linear(10, 5)`: input `[B, 10]`, output `[B, 5]`
    2. `nn.Embedding(100, 32)`: input `[B, T]` (long), output `[B, T, 32]`
    3. `nn.Conv1d(16, 32, kernel_size=3, padding=1)`: input `[B, 16, L]`, output `[B, 32, L]`
    4. `nn.Conv2d(3, 64, kernel_size=3, padding=1)`: input `[B, 3, H, W]`, output `[B, 64, H, W]`
    5. `nn.MaxPool2d(2, 2)`: input `[B, C, H, W]`, output `[B, C, H/2, W/2]`
    6. `nn.AdaptiveAvgPool2d((1, 1))`: input `[B, C, H, W]`, output `[B, C, 1, 1]`
    7. `nn.GRU(input_size=16, hidden_size=32, batch_first=True)`: input `[B, T, 16]`, output `[B, T, 32]`
    8. `nn.BatchNorm2d(64)`: input `[B, 64, H, W]`, output `[B, 64, H, W]`
    9. `nn.LayerNorm(32)`: input `[B, T, 32]`, output `[B, T, 32]`
- **Pass Criteria**:
  - Assert all expected shapes pass without error.

#### Assignment 4.4 — Custom Stateful Buffer Implementation
- **Objective**: Implement a module buffer for persistent non-parameter state (e.g. running class frequency counter).
- **Given**: Classification task with $C = 10$ classes.
- **Build / Do**:
  1. Build a module with `self.register_buffer("class_counts", torch.zeros(10, dtype=torch.long))`.
  2. In `forward(self, x, targets=None)`, if `targets` is provided during training, update counts under `torch.no_grad()` using `torch.bincount`.
  3. Test `.to(device)` placement, `state_dict()` inclusion, and optimizer interaction.
- **Pass Criteria**:
  - Assert that `class_counts` is present in `model.state_dict()`.
  - Assert that `class_counts` is **not** present in `list(model.parameters())`.
  - Assert that saving and loading `state_dict` correctly restores the accumulated counts.

#### Assignment 4.5 — Loss Contract & Target Format Laboratory
- **Objective**: Systematically document and test the input/target contracts and error modes of primary loss functions.
- **Given**:
  - Regression: `nn.MSELoss()`
  - Multi-class: `nn.CrossEntropyLoss()`
  - Binary/Multi-label: `nn.BCEWithLogitsLoss()`
- **Build / Do**:
  1. For each loss function, provide valid synthetic prediction and target tensors and compute loss.
  2. Deliberately test invalid inputs:
     - `CrossEntropyLoss`: pass float targets when expecting class indices; pass 1D logits.
     - `BCEWithLogitsLoss`: pass integer targets instead of float; pass mismatched shapes `[B, 1]` and `[B]`.
  3. Tabulate the contract: input shape/dtype, target shape/dtype, valid range, and reduction behavior.
- **Pass Criteria**:
  - Successfully trigger and document each expected error/warning or unintended broadcast.
  - Complete contract table accurately reflects PyTorch documentation.

#### Assignment 4.6 — Numerically Stable Classification Losses
- **Objective**: Demonstrate catastrophic numerical cancellation when applying softmax before cross-entropy.
- **Given**: Extreme logit vectors containing large positive and negative values (e.g. `logits = torch.tensor([[1000.0, -1000.0]])`, target `[0]`).
- **Build / Do**:
  1. Compute loss using the correct API: `loss_official = F.cross_entropy(logits, target)`.
  2. Compute loss using naive manual formulation: `probs = F.softmax(logits, dim=-1); loss_naive = -torch.log(probs[0, target])`.
  3. Compute loss using stable log-sum-exp formulation: `log_probs = F.log_softmax(logits, dim=-1); loss_stable = -log_probs[0, target]`.
- **Pass Criteria**:
  - Show that `loss_naive` evaluates to `inf` or `nan` due to floating point underflow in `softmax`.
  - Show that `loss_official` and `loss_stable` produce identical, finite, and accurate float values ($0.0$).
  - Explain why neural network classification heads must always output raw logits rather than probabilities.

**Exit criterion:** You can explain exactly what is stored in `parameters()`, `buffers()`, and `state_dict()`.

---

## Chapter 5 — Data pipelines: Dataset, sampler, batch, collate

### What are we learning in this chapter?
- **The Assembly Line of Data**: How data travels from raw files on your hard drive to GPU memory through four distinct components: `Dataset`, `Sampler`, `collate_fn`, and `DataLoader`.
- **Custom Map-Style Datasets**: Writing robust datasets using `__len__` and `__getitem__` with boundary validation.
- **Variable-Length Sequence Collation**: Writing custom `collate_fn` routines that pad unequal sequences and generate attention masks.
- **Data Loading Performance**: Tuning `num_workers`, `persistent_workers=True`, and `pin_memory=True` to ensure your GPU never starves.
- **Data Leakage Prevention**: Why preprocessing statistics (means, standard deviations, vocabularies) must strictly be computed on training splits.

---

### 1. The Four Responsibilities: An Assembly Line

When training a neural network, your GPU can process thousands of images or tokens per second. If your data loader is slow, your expensive GPU will sit idle, waiting for the CPU to load files.

PyTorch divides data loading into four clean, separated jobs:

```
+-------------------------------------------------------------------+
|                        1. Dataset                                 |
|  "How do I fetch ONE sample from disk or memory?"                 |
|  Implements: __len__() and __getitem__(idx)                       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        2. Sampler                                 |
|  "In what ORDER should I fetch the sample indices?"               |
|  Examples: SequentialSampler, RandomSampler, DistributedSampler   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       3. collate_fn                               |
|  "How do I pack a LIST of individual samples into a BATCH tensor?"|
|  Stacks tensors, pads variable lengths, creates masks             |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       4. DataLoader                               |
|  "The Factory Manager: Spawns worker processes, coordinates       |
|   background loading, and pins memory for fast GPU transfer"      |
+-------------------------------------------------------------------+
```

---

### 2. Custom Map-Style Dataset: Validate at the Boundary

A map-style dataset only needs two methods:
- `__len__(self)`: How many total samples exist?
- `__getitem__(self, idx)`: Return the sample at index `idx`.

> [!TIP]
> **Boundary Validation Rule:**
> Always validate shapes, dtypes, and non-finite values (`NaN`, `Inf`) inside `__getitem__`! Catching a corrupted image or missing value at the dataset boundary gives you an instant, helpful error naming the exact corrupt file. If you let it pass into the model, it will silently poison your loss with `NaN` 5 hours into training.

```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets), "Data and targets must have matching lengths"
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        # Validate boundary contract
        assert torch.isfinite(x).all(), f"NaN/Inf found in sample {idx}"
        return x, y
```

---

### 3. Collation: Packaging Batches with Variable Lengths

By default, PyTorch's `DataLoader` uses `default_collate`, which takes a list of sample tuples and stacks them along dimension 0.
This works great if all images or vectors have identical shapes!

**What if your sequences have different lengths?**
Suppose sample 1 has 3 words, sample 2 has 5 words, and sample 3 has 2 words.
You cannot stack `[3, D]` and `[5, D]` into a rectangular tensor!
You must write a custom `collate_fn`:

```python
def pad_collate_fn(batch):
    # batch is a list of tuples: [(tokens_1, label_1), (tokens_2, label_2), ...]
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # 1. Pad sequences to match the longest sequence in this batch
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # 2. Build a boolean padding mask (True for real tokens, False for padding)
    mask = (padded_seqs != 0)
    
    return {"tokens": padded_seqs, "mask": mask, "labels": labels}
```

---

### 4. DataLoader Tuning: Keeping the GPU Fed

When training on a GPU, use these three settings:
1. `num_workers=4`: Uses 4 separate CPU background processes to load and preprocess data in advance while the GPU is busy computing.
2. `pin_memory=True`: Allocates loaded tensors in page-locked ("pinned") host RAM. This allows the GPU to copy memory directly via Direct Memory Access (DMA) without CPU involvement, dramatically speeding up `.to("cuda")`.
3. `persistent_workers=True`: Keeps background worker processes alive between epochs instead of destroying and respawning them every epoch.

---

### 5. Collation Under the Hood: `default_collate`

What does PyTorch do when you don't provide a custom `collate_fn`?
It uses `torch.utils.data.dataloader.default_collate`.
`default_collate` inspects the structure of your samples recursively:
- **Tensors:** Stacks them along dimension 0 (`torch.stack`).
- **Python numbers (int/float):** Automatically converts them into a 1D tensor.
- **Dictionaries:** Returns a single dictionary with the same keys, where each value is a batched tensor of the stacked items across samples!

```python
from torch.utils.data.dataloader import default_collate

raw_samples = [
    {"id": 0, "x": torch.tensor([1.0, 2.0]), "label": 0},
    {"id": 1, "x": torch.tensor([3.0, 4.0]), "label": 1},
]

batch = default_collate(raw_samples)
# batch["id"] -> tensor([0, 1])
# batch["x"]  -> tensor([[1.0, 2.0], [3.0, 4.0]])
# batch["label"] -> tensor([0, 1])
```

---

### 6. Data Transforms: Stochastic Training vs Deterministic Evaluation

Data preprocessing must always be split into two separate pipelines:
1. **Training Transforms (Stochastic):**
   Applies random perturbations (e.g. random horizontal flip, random crop, color jitter) to teach the model to recognize objects regardless of orientation or lighting.
2. **Evaluation Transforms (Deterministic):**
   Applies fixed, reproducible preprocessing (e.g. resize, center crop, normalize). Never apply random transformations during evaluation, or your test score will fluctuate randomly!

```python
def train_transform(img):
    # Random horizontal flip (stochastic)
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)
    # Normalize
    return (img - 0.5) / 0.5

def eval_transform(img):
    # Deterministic: no random flipping!
    return (img - 0.5) / 0.5
```

---

### 7. Streaming Large Datasets with `IterableDataset` & Multi-Worker Safety

When your dataset is too massive to fit in memory or is streamed over a network, subclass `IterableDataset`:
- Instead of `__getitem__(idx)`, implement `__iter__(self)`.

> [!CAUTION]
> **The Multi-Worker Duplication Bug:**
> If you create a `DataLoader` with `num_workers=2` on an `IterableDataset`, PyTorch will spawn 2 independent worker processes running `__iter__()`. If you don't partition the stream, **both workers will read the entire dataset**, duplicating every sample!

**The Fix:** Partition the stream using `torch.utils.data.get_worker_info()`:
```python
from torch.utils.data import IterableDataset, DataLoader

class RangeStream(IterableDataset):
    def __init__(self, total_items):
        self.total_items = total_items
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process mode: yield all items
            yield from range(self.total_items)
        else:
            # Multi-worker mode: partition by striding across workers!
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            for item in range(worker_id, self.total_items, num_workers):
                yield item
```

---

### 8. Data Leakage: Why Preprocessing Statistics Belong Only to Training

**Data leakage** occurs when information from outside the training dataset is used to create the model.
- **The Rookie Mistake:** Normalizing your entire dataset (training + validation together) by computing the global mean and standard deviation:
  $$\mu = \text{mean}(X_{\text{train}} \cup X_{\text{val}})$$
- **Why this is fatal:** In the real world, you cannot see future test data. If your validation set contains extreme outliers or shifted values, computing statistics on the whole dataset leaks validation distribution information into training!
- **The Law:** Compute $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ **solely on the training split**. Then use those exact same fixed values to normalize the validation and test splits.

---

### Assignments 5

#### Assignment 5.1 — Trace Default Collation Mechanics
- **Objective**: Understand how PyTorch's default collator transforms a list of individual sample dictionaries into a batched dictionary.
- **Given**: A custom dataset where each sample is a dictionary:
  ```python
  sample = {"id": int(i), "x": torch.randn(4), "label": torch.tensor(i % 2, dtype=torch.long)}
  ```
- **Build / Do**:
  1. Instantiate a list of 3 raw samples: `raw_batch = [dataset[0], dataset[1], dataset[2]]`.
  2. Pass `raw_batch` through `torch.utils.data.dataloader.default_collate`.
  3. Inspect the structure, types, and shapes of the resulting object.
  4. Write a manual collation function using `torch.stack` and verify it produces identical output.
- **Pass Criteria**:
  - Assert that `batch["id"]` is a 1D tensor of shape `(3,)` (integers converted to tensors).
  - Assert that `batch["x"]` has shape `(3, 4)`.
  - Assert that `batch["label"]` has shape `(3,)`.
  - Manual collate output passes `torch.testing.assert_close` against default collate.

#### Assignment 5.2 — Validated CSV Dataset with Error Reporting
- **Objective**: Build a robust map-style CSV dataset that validates data integrity and provides actionable error messages.
- **Given**: A CSV file format with header `f1,f2,f3,label`. Prepare both valid CSV rows and synthetic corrupt rows (missing column, non-numeric feature, non-integer label, NaN/Inf value, out-of-range label $> 2$).
- **Build / Do**:
  1. Implement `ValidatedCSVDataset(csv_path)` using Python's `csv` module.
  2. In `__getitem__(idx)`, parse and validate row data:
     - Check exact column count.
     - Convert `[f1, f2, f3]` to `torch.float32`; verify all values are finite (`torch.isfinite`).
     - Convert `label` to `torch.long`; verify $0 \le \text{label} \le 2$.
  3. If invalid, raise a `ValueError` identifying the exact row number, column name, and invalid value.
- **Pass Criteria**:
  - Valid rows return tuple `(torch.Tensor[3], torch.Tensor[])`.
  - Every corrupted test case raises `ValueError` containing row index and problem description.

#### Assignment 5.3 — Variable-Length Sequence Collation with Padding
- **Objective**: Implement a custom `collate_fn` to batch sequences of varying lengths with padding and attention masks.
- **Given**: Three variable-length token sequences:
  - Sequence 1: `torch.tensor([5, 8], dtype=torch.long)` (length 2)
  - Sequence 2: `torch.tensor([4], dtype=torch.long)` (length 1)
  - Sequence 3: `torch.tensor([9, 2, 7], dtype=torch.long)` (length 3)
  - Labels: `[0, 1, 0]`, Padding token ID: `0`.
- **Build / Do**:
  1. Write `pad_collate_fn(batch)`:
     - Pad sequences to the maximum length in the batch ($3$) using `torch.nn.utils.rnn.pad_sequence(..., batch_first=True, padding_value=0)`.
     - Construct a boolean padding mask of shape `[3, 3]` (`mask = (padded_tokens != 0)`).
     - Return dictionary: `{"tokens": padded_tokens, "lengths": lengths, "mask": mask, "labels": labels}`.
  2. Implement a function that calculates the mean feature vector over valid (non-padding) tokens only.
- **Pass Criteria**:
  - Assert `padded_tokens` has shape `(3, 3)` with padded values:
    - Row 1: `[5, 8, 0]`
    - Row 2: `[4, 0, 0]`
    - Row 3: `[9, 2, 7]`
  - Assert `mask` has shape `(3, 3)` and dtype `torch.bool`.
  - Assert non-padding mean ignores 0-padding values accurately.

#### Assignment 5.4 — Transform Pipeline Safety & Determinism
- **Objective**: Ensure proper separation between stochastic training augmentations and deterministic evaluation transforms.
- **Given**: A test image tensor `img = torch.rand(3, 64, 64)`.
- **Build / Do**:
  1. Build a training transform pipeline containing random horizontal flip, random cropping, and normalization.
  2. Build an evaluation transform pipeline containing center crop and identical normalization (no stochastic augmentation).
  3. Apply the training pipeline 5 times to `img` and record outputs.
  4. Apply the evaluation pipeline 5 times to `img` and record outputs.
- **Pass Criteria**:
  - Assert that all 5 evaluation outputs are strictly identical: `torch.equal(eval_out[0], eval_out[i])`.
  - Assert that training outputs exhibit stochastic variation: `not torch.equal(train_out[0], train_out[1])`.
  - Assert that both pipelines output identical shape `(3, 56, 56)` and dtype `torch.float32`.

#### Assignment 5.5 — Multi-Worker Safe Stream Dataset
- **Objective**: Implement an `IterableDataset` that partitions data across workers without duplication or omissions.
- **Given**: A synthetic data stream of integers from $0$ to $99$ (`RangeStream(100)`).
- **Build / Do**:
  1. Implement `RangeStream(IterableDataset)`:
     - In `__iter__()`, query `torch.utils.data.get_worker_info()`.
     - If `worker_info is None` (single process), yield all 100 items.
     - If running in multi-worker mode, partition the range based on `worker_info.id` and `worker_info.num_workers` (e.g. stride or slice).
  2. Load with `DataLoader` using `num_workers=0`, `num_workers=2`, and `num_workers=4`.
- **Pass Criteria**:
  - For all worker configurations, collect all yielded numbers across one full epoch.
  - Assert `sorted(collected) == list(range(100))` with no duplicate elements and no dropped elements.

#### Assignment 5.6 — DataLoader Performance Benchmarking
- **Objective**: Empirically profile data loading configurations to select the optimal worker and memory settings.
- **Given**: A synthetic or real dataset of 10,000 samples with batch size 64.
- **Build / Do**:
  1. Benchmark throughput across combinations:
     - `num_workers` $\in \{0, 1, 2, 4\}$
     - `pin_memory` $\in \{\text{False}, \text{True}\}$ (if GPU is available)
     - `persistent_workers` $\in \{\text{False}, \text{True}\}$
  2. Measure median examples per second after a 1-epoch warmup across 3 measured epochs.
- **Pass Criteria**:
  - Tabulate throughput (examples/sec) for each configuration.
  - Written justification for the chosen configuration based on measured data rather than default assumptions.

#### Assignment 5.7 — Data Leakage Demonstration
- **Objective**: Demonstrate how fitting preprocessing statistics across the entire dataset leaks validation information.
- **Given**:
  - Seed: `torch.manual_seed(0)`
  - Training set: `train_data = torch.randn(100, 1)` ($\mu \approx 0$)
  - Validation set: `val_data = 8.0 + torch.randn(100, 1)` ($\mu \approx 8$)
- **Build / Do**:
  1. Pipeline A (Correct): Fit mean and standard deviation solely on `train_data`; apply fitted statistics to normalize `val_data`.
  2. Pipeline B (Leaked): Concatenate `train_data` and `val_data`; fit mean and standard deviation on the combined dataset; apply to `val_data`.
  3. Shift validation values by adding $+10.0$ and re-run both pipelines.
- **Pass Criteria**:
  - Show that in Pipeline A, the fitted mean depends only on training data and remains unchanged when validation data shifts.
  - Show that in Pipeline B, the fitted mean changes when validation data shifts, proving information leakage.

**Exit criterion:** You can trace one sample from raw storage through dataset retrieval, collation, worker transfer, and model input.

---

## Chapter 6 — The training loop, correctly

### What are we learning in this chapter?
- **The Canonical 7-Step Training Step**: The exact, non-negotiable step order for training models in PyTorch.
- **The Evaluation Loop**: How to correctly evaluate models without leaking gradients or corrupting BatchNorm stats.
- **Weighted Loss Aggregation**: Why simply averaging batch loss numbers gives the wrong answer when batches are uneven, and how to weight by sample count.
- **Optimizers Demystified**: How SGD with momentum works, and why AdamW (decoupled weight decay) outperforms classic Adam.
- **Learning Rate Schedulers**: Exactly when to call `scheduler.step()` (batch-based vs. epoch-based vs. metric-based).
- **Gradient Accumulation**: Simulating large batch sizes on small GPUs with exact mathematical parity.
- **Bulletproof Checkpoint Resume**: What must be saved (model, optimizer, scheduler, epoch, RNG states) so you can resume training seamlessly.
- **The 32-Sample Overfit Proof**: The ultimate diagnostic test to verify your pipeline before running long experiments.

---

### 1. The Canonical 7-Step Training Loop

Every single training step in PyTorch follows this exact sequence:

```python
# 1. Set model to training mode (activates Dropout and BatchNorm updates)
model.train()

for batch_idx, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    
    # 2. Forward pass: compute predictions
    pred = model(x)
    
    # 3. Compute loss
    loss = criterion(pred, y)
    
    # 4. Zero out accumulated gradients from previous step
    optimizer.zero_grad(set_to_none=True)
    
    # 5. Backward pass: compute gradients via autograd
    loss.backward()
    
    # 6. Optional: Clip exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 7. Step the optimizer: update parameters using gradients
    optimizer.step()
```

---

### 2. The Evaluation Loop: Mode and Context

To evaluate your model on validation data, you must do two separate things:
1. **`model.eval()`:** Changes layer behavior (turns off Dropout; freezes BatchNorm running stats).
2. **`with torch.inference_mode():`:** Disables gradient computation and graph construction to save memory and run faster.

```python
model.eval()
total_loss = 0.0
total_samples = 0

with torch.inference_mode():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        
        # Accumulate loss weighted by batch size
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

val_loss = total_loss / total_samples
```

---

### 3. Optimizers: SGD vs. AdamW

- **SGD (Stochastic Gradient Descent):**
  - Moves weights in the exact direction of the gradient: $w \leftarrow w - \eta \nabla L$.
  - With **momentum**, it adds velocity from previous steps (like a heavy ball rolling downhill), helping it escape shallow local minima and flat regions.
- **AdamW (Adaptive Moment Estimation with Decoupled Weight Decay):**
  - Maintains individual learning rates for every parameter based on past gradient magnitudes.
  - Fixes the bug in original Adam where $L_2$ regularization was blended into gradient moments instead of directly decaying the weights.

---

### 4. Learning Rate Schedulers: Step Timing

Calling `scheduler.step()` at the wrong time will silently ruin your training schedule:
1. **Batch Schedulers (e.g. `CosineAnnealingLR`, `OneCycleLR`):**
   - Must be called **inside the batch loop**, immediately after `optimizer.step()`.
2. **Epoch Schedulers (e.g. `StepLR`, `MultiStepLR`):**
   - Must be called **at the end of each epoch**, outside the batch loop.
3. **Metric Schedulers (e.g. `ReduceLROnPlateau`):**
   - Must be called **after validation**, passing the validation metric: `scheduler.step(val_loss)`.

---

### 5. Checkpointing: Bit-for-Bit Exact Resumption

If training crashes on step 10,000, you don't want to restart from step 0.
To resume training *exactly* as if no interruption occurred, you must save:
1. `model.state_dict()` (weights and buffers)
2. `optimizer.state_dict()` (momentum buffers and learning rates)
3. `scheduler.state_dict()` (current step in schedule)
4. `scaler.state_dict()` (if using mixed precision AMP)
5. `epoch` and `best_metric`
6. `torch.get_rng_state()` (PyTorch random generator state)

```python
# Saving
checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "rng_state": torch.get_rng_state(),
}
torch.save(checkpoint, "checkpoint.pt")

# Resuming
checkpoint = torch.load("checkpoint.pt", weights_only=False)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
torch.set_rng_state(checkpoint["rng_state"])
start_epoch = checkpoint["epoch"] + 1
```

---

### 6. Gradient Accumulation: Simulating Huge Batches on Small GPUs

If your GPU runs out of memory when training with batch size 64, you can simulate batch size 64 by running 4 microbatches of size 16!

**The Mathematical Equivalence:**
Because gradients add up:
$$\nabla L_{\text{total}} = \frac{1}{4} \nabla L_1 + \frac{1}{4} \nabla L_2 + \frac{1}{4} \nabla L_3 + \frac{1}{4} \nabla L_4$$

**The Implementation:**
```python
accumulation_steps = 4

for i, (x, y) in enumerate(train_loader):
    pred = model(x)
    # Scale loss down by accumulation_steps so gradients aren't 4x too large!
    loss = criterion(pred, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```
> [!NOTE]
> If microbatches have unequal sizes (e.g., sizes 4, 4, and 2 for a total of 10), each microbatch loss must be scaled by its actual sample count divided by total sample count: `loss * (batch_size / total_samples)`.

---

### 7. Diagnosing the 4 Silent Training Pipeline Bugs

These four silent bugs do not crash your program, but will prevent your model from learning:

1. **Applying Softmax Before `CrossEntropyLoss`:**
   - *Symptom:* Loss plateaus or gradient vanishes.
   - *Fix:* Remove softmax. Pass raw logits directly to `CrossEntropyLoss`.

2. **Omitting `model.eval()` During Validation:**
   - *Symptom:* Validation accuracy fluctuates randomly and is much worse than training accuracy.
   - *Root Cause:* Dropout keeps dropping 50% of your neurons at test time, and BatchNorm keeps recalculating mean/var on tiny test batches!
   - *Fix:* Always set `model.eval()` before validation, and `model.train()` before training.

3. **Omitting `optimizer.zero_grad()`:**
   - *Symptom:* Loss explodes to `NaN` or `Inf` within 5 steps.
   - *Root Cause:* Gradients keep accumulating indefinitely on every step!
   - *Fix:* Call `optimizer.zero_grad(set_to_none=True)` before `loss.backward()`.

4. **Creating the Optimizer Before Modifying Model Architecture:**
   - *Symptom:* Head weights never update; model output never improves.
   - *Root Cause:* If you write `optimizer = Adam(model.parameters())` and THEN replace `model.fc = nn.Linear(...)`, the optimizer is still tracking the *old, discarded* layer parameters!
   - *Fix:* Always modify the model architecture FIRST, then instantiate the optimizer.

---

### Assignments 6

#### Assignment 6.1 — Complete Classifier with Exact Metric Aggregation
- **Objective**: Build a complete, robust training and validation loop for a 2D binary classification task.
- **Given**:
  - Synthetic dataset: two concentric noisy circles or 2D moons with 1,000 samples.
  - Deterministic 80/20 train/validation split.
  - Architecture: MLP `2 -> 32 -> 2` with ReLU activations.
- **Build / Do**:
  1. Train for 50 epochs with `AdamW(lr=1e-2)`.
  2. Implement proper epoch loss aggregation:
     $$\text{Epoch Loss} = \frac{\sum_{\text{batches}} \text{batch\_loss} \times \text{batch\_size}}{N_{\text{total}}}$$
  3. Track per-epoch train loss, validation loss, train accuracy, and validation accuracy.
- **Pass Criteria**:
  - Validation accuracy exceeds majority baseline ($50\%$) by at least $25\%$ ($> 75\%$).
  - Code explicitly demonstrates weighted batch loss aggregation.

#### Assignment 6.2 — Tiny-Subset Capacity Proof
- **Objective**: Prove model implementation correctness by overfitting a tiny batch to near-zero loss.
- **Given**: Exactly 32 samples extracted from a dataset, with augmentation disabled.
- **Build / Do**:
  1. Train the model on this fixed 32-sample batch for up to 200 iterations.
  2. Monitor training loss and accuracy.
- **Pass Criteria**:
  - Model achieves at least 31/32 ($> 96.8\%$) accuracy and loss $< 0.05$.
  - If it fails, follow the systematic diagnostic checklist: data format $\to$ shape consistency $\to$ loss formulation $\to$ gradient existence $\to$ parameter update.

#### Assignment 6.3 — Optimizer Trajectory Comparison
- **Objective**: Empirically compare convergence trajectories between SGD, SGD with momentum, and AdamW.
- **Given**: Fixed synthetic regression or classification dataset with identical initial model weights.
- **Build / Do**:
  1. Run 100 optimization steps for:
     - Plain SGD (`lr=0.01`, `momentum=0.0`)
     - SGD with momentum (`lr=0.01`, `momentum=0.9`)
     - AdamW (`lr=0.001`, `weight_decay=1e-2`)
  2. Record loss at every step.
- **Pass Criteria**:
  - Plot or tabulate loss vs. step for all three optimizers.
  - Provide written explanation of why default hyperparameters do not constitute a fair universal ranking of optimizers.

#### Assignment 6.4 — Learning Rate Scheduler Step Timing
- **Objective**: Implement and verify correct step timing for epoch-based, batch-based, and metric-based schedulers.
- **Given**: A small model and dataset trained for 5 epochs of 10 batches each (50 total updates).
- **Build / Do**:
  1. Run 1: `StepLR(optimizer, step_size=2, gamma=0.5)` stepped once per epoch after validation.
  2. Run 2: `CosineAnnealingLR(optimizer, T_max=50)` stepped once per batch after `optimizer.step()`.
  3. Run 3: `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)` stepped once per epoch with validation loss.
  4. Log learning rate at every optimizer update: `optimizer.param_groups[0]['lr']`.
- **Pass Criteria**:
  - Verify that `StepLR` steps exactly at epochs 2 and 4.
  - Verify that `CosineAnnealingLR` updates smoothly at all 50 batches.
  - Verify that `ReduceLROnPlateau` steps only when validation loss fails to improve for `patience` epochs.

#### Assignment 6.5 — Exact Checkpoint Save and Resume
- **Objective**: Prove that saving and resuming training produces identical results to uninterrupted training.
- **Given**: Fixed CPU training setup with `num_workers=0` and fixed seed.
- **Build / Do**:
  1. Reference run: Train uninterrupted for 5 epochs; record final weights, optimizer state, and loss.
  2. Checkpoint run:
     - Train for 3 epochs.
     - Save checkpoint containing: `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, `epoch`, and `torch.get_rng_state()`.
     - Terminate process / destroy objects.
     - Reconstruct model, optimizer, scheduler.
     - Load checkpoint, restore RNG state, and train for remaining 2 epochs.
- **Pass Criteria**:
  - Assert that parameters from resumed run match uninterrupted reference:
    `torch.testing.assert_close(resumed_model.parameters(), ref_model.parameters(), rtol=1e-5, atol=1e-6)`.
  - Assert that minibatch order and learning rates in epochs 4 and 5 match exactly.

#### Assignment 6.6 — Mathematical Equivalence of Gradient Accumulation
- **Objective**: Prove that gradient accumulation over $N$ microbatches is mathematically equivalent to one large batch.
- **Given**: Initial model with disabled Dropout and BatchNorm (or fixed `eval()` mode).
- **Build / Do**:
  1. Baseline: Compute forward, loss, and update for one full batch of 16 samples using SGD (`lr=0.01`).
  2. Accumulated: From identical initial weights, split the 16 samples into 4 microbatches of 4.
     - For each microbatch, compute `loss = criterion(pred, target) / 4`.
     - Call `loss.backward()`.
     - Call `optimizer.step()` only after all 4 microbatches, then zero gradients.
  3. Repeat with unequal microbatches (e.g. sizes 4, 4, 2 for total batch of 10) and verify normalization by 10.
- **Pass Criteria**:
  - Assert `torch.testing.assert_close` between baseline and accumulated parameters with `rtol=1e-5, atol=1e-6`.
  - Assert that the unequal batch update normalizes by total sample count (10) rather than microbatch count (3).

#### Assignment 6.7 — Training Failure Signatures & Diagnosis
- **Objective**: Identify, diagnose, and fix the 4 most common silent training pipeline bugs.
- **Given**: Four deliberately corrupted training setups:
  1. Applying `F.softmax` immediately before `nn.CrossEntropyLoss`.
  2. Omitting `model.eval()` during validation (with active Dropout and BatchNorm).
  3. Omitting `optimizer.zero_grad()` inside the training loop.
  4. Creating the optimizer before replacing the final classification layer.
- **Build / Do**:
  1. Run each broken pipeline for 5 steps.
  2. Document the visible symptom (e.g. gradient accumulation explosion, poor eval accuracy, unchanged head weights).
  3. Implement the minimal one-line correction for each case.
- **Pass Criteria**:
  - Provide a diagnostic report detailing: Symptom, Diagnostic Check, Root Cause, and Minimal Fix for each bug.

**Exit criterion:** From an empty file, you can build, validate, save, load, and resume a complete training loop.

---

## Chapter 7 — Reproducibility, tests, and experiment discipline

### What are we learning in this chapter?
- **The Sources of Randomness**: Where stochasticity lives (Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA).
- **The Universal `seed_everything` Pattern**: Writing a single function that locks down all random number generators.
- **Enforcing Deterministic Algorithms**: Using `torch.use_deterministic_algorithms(True)` to catch non-deterministic CUDA operations.
- **Automated ML Unit Tests**: Writing automated smoke tests for shape contracts, gradient propagation, and evaluation invariance.
- **Scientific Experiment Discipline**: Tracking git commits, hyperparameters, and environment metadata alongside training logs.

---

### 1. Where Does Randomness Live?

If you run your training script twice and get two completely different results, you cannot know whether an improvement was caused by your clever idea or just a lucky random seed.

Randomness enters PyTorch code from four separate libraries:
1. Python's built-in `random` module (used for list shuffling).
2. NumPy's `numpy.random` (used for data augmentations).
3. PyTorch CPU generator (`torch.manual_seed`).
4. PyTorch GPU generators (`torch.cuda.manual_seed_all`).

Here is the bulletproof `seed_everything` utility every ML engineer should use:
```python
import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force cuDNN to use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

#### Saving and Restoring Full RNG State Checkpoints
While `seed_everything()` resets generators at the beginning of a run, what if your training job is paused or interrupted mid-epoch?
If you simply re-seed at step 0, your resumed job will repeat the exact same sequence of random operations that already ran, causing data duplication!
To resume training seamlessly, capture and restore the **current internal generator state** of all libraries:

```python
# 1. Capture current RNG states across all random sources:
rng_state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
}

# 2. Restore RNG states when resuming:
random.setstate(rng_state["python"])
np.random.set_state(rng_state["numpy"])
torch.set_rng_state(rng_state["torch_cpu"])
if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
    torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
```
This guarantees bit-for-bit identical behavior between an uninterrupted run and a checkpoint-resumed run.

---

### 2. Enforcing Deterministic CUDA Algorithms

Some GPU algorithms (like certain atomic additions in backward convolutions or `scatter_add`) sacrifice bit-for-bit determinism for maximum speed.
If your project requires exact repeatability, enforce determinism:
```python
torch.use_deterministic_algorithms(True)
```
If using CUDA, you must also set this environment variable before launching Python:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
If an operation does not have a deterministic implementation, PyTorch will raise a `RuntimeError` immediately, telling you exactly which operation is non-deterministic.

---

### 3. Essential ML Unit Tests

Before training a model for days, write three simple unit tests:
1. **Shape Smoke Test:** Pass a dummy batch `torch.randn(2, C, H, W)` and assert the output shape matches `[2, NumClasses]`.
2. **Parameter Update Test:** Take one step with dummy data and assert that every parameter where `requires_grad=True` actually changed value. If a weight didn't change, its gradient path is broken!
3. **Eval Invariance Test:** Call `model.eval()`, pass the same input twice, and assert the two outputs are 100% identical. If they differ, you forgot to disable a stochastic layer!

---

### Assignments 7

#### Assignment 7.1 — Automated ML Unit Smoke Test Suite
- **Objective**: Build an automated test suite verifying layer shapes, gradient flow, and parameter updates.
- **Given**: A custom neural network module.
- **Build / Do**:
  1. Write `test_forward_shape()`: pass dummy batch of shape `[2, C, H, W]` and assert expected output shape `[2, NumClasses]`.
  2. Write `test_all_parameters_updated()`:
     - Clone initial parameters.
     - Execute one forward, loss, backward, and `optimizer.step()`.
     - Assert that every parameter in `model.parameters()` where `requires_grad=True` has changed:
       `assert not torch.equal(param, initial_param)`.
  3. Write `test_eval_invariance()`:
     - Call `model.eval()`.
     - Pass the same fixed input twice and assert `torch.equal(output1, output2)`.
- **Pass Criteria**:
  - All three tests pass via `pytest` or a standalone test script.

#### Assignment 7.2 — Multi-Source Random Seed Isolation
- **Objective**: Ensure complete reproducibility across Python, NumPy, and PyTorch random streams.
- **Given**: Data pipeline combining Python `random.shuffle`, NumPy `np.random.rand`, and PyTorch `torch.randn`.
- **Build / Do**:
  1. Write a seeding utility `seed_everything(seed=42)` setting seeds for Python, NumPy, and PyTorch (CPU + all GPUs).
  2. Run the pipeline twice from `seed_everything(42)`.
  3. Run the pipeline once from `seed_everything(43)`.
- **Pass Criteria**:
  - Assert that all outputs from the two seed 42 runs are strictly identical (`torch.equal`).
  - Assert that outputs from seed 43 differ from seed 42.

#### Assignment 7.3 — Complete RNG State Checkpointing
- **Objective**: Capture and restore full RNG state across multiple libraries.
- **Given**: A process drawing successive random numbers from PyTorch CPU, CUDA (if available), and Python `random`.
- **Build / Do**:
  1. Seed at step 0; draw 5 random numbers.
  2. Capture RNG state dictionary:
     ```python
     rng_state = {
         "torch_cpu": torch.get_rng_state(),
         "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
         "python": random.getstate(),
     }
     ```
  3. Draw 5 more numbers (Run A).
  4. Restore `rng_state` and draw 5 numbers (Run B).
- **Pass Criteria**:
  - Assert that all 5 numbers in Run B match Run A bit-for-bit.

#### Assignment 7.4 — Deterministic Algorithm Audit
- **Objective**: Identify non-deterministic CUDA operations and enforce deterministic execution.
- **Given**: Operations that historically use non-deterministic CUDA kernels (e.g. `index_add_`, `scatter_add_`, or certain backward convolutions).
- **Build / Do**:
  1. Enable `torch.use_deterministic_algorithms(True)`.
  2. If using CUDA, set environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`.
  3. Execute the candidate operations.
- **Pass Criteria**:
  - Code completes without raising `RuntimeError: ... does not have a deterministic implementation` or catches and documents the unsupported operation.

**Exit criterion:** Every experiment you run is 100% reproducible and protected by automated smoke tests.

---

# Part III — Architectures

## Chapter 8 — Convolutional networks

### What are we learning in this chapter?
- **Why Convolutions for Images**: Why dense (fully-connected) layers fail on images (millions of redundant parameters, loss of spatial structure) and how weight sharing fixes it.
- **The 2D Convolution Operation from Scratch**: Understanding kernels, strides, padding, and dilation with clear grid visuals.
- **The Output Spatial Dimension Formula**: Deriving and mastering the formula that predicts feature map height and width.
- **Pooling & Resolution Independence**: Using `nn.MaxPool2d` for downsampling and `nn.AdaptiveAvgPool2d((1, 1))` to handle variable image resolutions.
- **Batch Normalization & Residual Connections**: How `nn.BatchNorm2d` stabilizes intermediate activations and how skip connections ($F(x) + x$) solve the vanishing gradient problem in deep networks.
- **Feature Inspection with Forward Hooks**: Extracting intermediate layer representations without modifying the model's forward method.

---

### 1. What problem does convolution solve?

Suppose you have a modest color image: 256 pixels tall, 256 pixels wide, with 3 color channels (RGB).
If you flatten that image into a 1D vector, it has:
$$256 \times 256 \times 3 = 196,608 \text{ numbers!}$$

If you connect this to a standard dense layer (`nn.Linear`) with 1,000 hidden units:
$$196,608 \times 1,000 = 196,608,000 \text{ weights!}$$
Almost 200 million parameters for just the first layer!
Worse still:
1. **It ignores spatial locality:** Nearby pixels are strongly related (e.g. pixels forming an eye or an edge). A dense layer scrambles all pixels into a flat list, treating pixels 1 millimeter apart the same as pixels on opposite corners of the image.
2. **It lacks translation equivariance:** If a cat appears in the top-left corner, a dense layer has to learn what a cat looks like from scratch in the bottom-right corner.

**The Convolution Solution:**
Instead of looking at the whole image at once, slide a tiny filter (called a **kernel**, typically $3 \times 3$) across the image.
- **Local connectivity:** The filter only looks at small local patches.
- **Weight sharing:** The *exact same* $3 \times 3$ filter is applied everywhere across the image. If it learns to detect an edge, it detects edges anywhere!
A $3 \times 3$ filter across 3 channels has only $3 \times 3 \times 3 = 27$ weights!

---

### 2. The 2D Convolution Operation: A Step-by-Step Grid

Think of a convolution kernel as a magnifying glass with pattern weights.
At each position on the image, the kernel computes a **dot-product**:
1. Multiply each kernel weight by the pixel value directly underneath it.
2. Sum all the products together.
3. Add a bias term to get a single output number.

```
INPUT (4x4)                  KERNEL (3x3)              OUTPUT (2x2)
+---+---+---+---+            +---+---+---+
| 1 | 2 | 0 | 1 |            | 1 | 0 |-1 |
+---+---+---+---+     *      +---+---+---+     -->     +----+----+
| 0 | 1 | 3 | 2 |            | 1 | 0 |-1 |             |-1  | 0  |
+---+---+---+---+            +---+---+---+             +----+----+
| 2 | 0 | 1 | 0 |            | 1 | 0 |-1 |             |-1  |-2  |
+---+---+---+---+            +---+---+---+             +----+----+
| 1 | 1 | 0 | 2 |
+---+---+---+---+
```

#### The Output Size Formula
How big is the output feature map?
$$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K - 1) - 1}{S} + 1 \right\rfloor$$
Where:
- $H_{in}$: Input height
- $P$: Padding (adding zeros around the border)
- $K$: Kernel size (e.g. 3 for $3 \times 3$)
- $S$: Stride (how many pixels the kernel shifts per step)
- $D$: Dilation (spacing between kernel elements, default 1)

*Example:* With $H_{in}=4, P=0, K=3, S=1, D=1$:
$$H_{out} = \frac{4 + 0 - 1(2) - 1}{1} + 1 = 2$$

---

### 3. Residual Connections: The Highway for Gradients

As neural networks get deeper (20, 50, 100 layers), gradients flowing backward get multiplied by weights over and over. They either vanish to zero (network stops learning) or explode to infinity.

The **Residual Connection** (ResNet) solves this with a simple addition:
$$\text{Output} = F(x) + x$$
Instead of forcing the layers $F(x)$ to learn the entire output from scratch, they only have to learn the *residual* (the difference/change).
During backpropagation, the $+ x$ shortcut creates an uninterrupted gradient highway:
$$\frac{\partial (F(x) + x)}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$$
Even if the gradient of $F(x)$ is zero, the $+1$ guarantees that gradients flow straight through to earlier layers!

---

### 4. Residual Blocks with Projection Shortcuts & BatchNorm Mechanics

#### A. Dimension-Matching Shortcuts ($1 \times 1$ Convolutions)
If a residual block downsamples the image (e.g. `stride=2`) or increases the channel count (e.g. 16 $\to$ 32), the shapes of $F(x)$ and $x$ no longer match! You cannot perform $F(x) + x$.
To fix this, pass $x$ through a **projection shortcut** using a $1 \times 1$ convolution:
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path:
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)
```

#### B. BatchNorm in `train()` vs `eval()` Mode
`nn.BatchNorm2d` behaves completely differently in the two modes:
1. **Training (`model.train()`):**
   - Computes batch mean $\mu_B$ and variance $\sigma^2_B$ across the current minibatch.
   - Updates internal persistent buffers `running_mean` and `running_var` using exponential moving average.
   - *Trap:* If you pass a batch of size 1 in `train()` mode, PyTorch raises a `ValueError` because you cannot compute sample variance on a single item!
2. **Evaluation (`model.eval()`):**
   - Freezes running statistics. Normalizes inputs using the stored `running_mean` and `running_var`.
   - Batch size 1 works perfectly during evaluation because the statistics are already known!

---

### Assignments 8

#### Assignment 8.1 — 2D Convolution by Hand and Verification
- **Objective**: Calculate 2D cross-correlation manually on paper and verify against `nn.Conv2d`.
- **Given**:
  - Input: $1 \times 1 \times 4 \times 4$ image with values:
    ```
    [[1, 2, 0, 1],
     [0, 1, 3, 2],
     [2, 0, 1, 0],
     [1, 1, 0, 2]]
    ```
  - Kernel: $1 \times 1 \times 3 \times 3$ with values:
    ```
    [[ 1,  0, -1],
     [ 1,  0, -1],
     [ 1,  0, -1]]
    ```
  - Stride: 1, Padding: 0, Bias: 0.
- **Build / Do**:
  1. Manually calculate the $2 \times 2$ output spatial values on paper.
  2. Implement in PyTorch using `F.conv2d` or `nn.Conv2d`.
- **Pass Criteria**:
  - Output values match manual derivation:
    - Position (0, 0): $(1 - 0) + (0 - 3) + (2 - 1) = 1 - 3 + 1 = -1$
    - Position (0, 1): $(2 - 1) + (1 - 2) + (0 - 0) = 1 - 1 + 0 = 0$
    - Position (1, 0): $(0 - 3) + (2 - 1) + (1 - 0) = -3 + 1 + 1 = -1$
    - Position (1, 1): $(1 - 2) + (0 - 0) + (1 - 2) = -1 + 0 - 1 = -2$
  - Assert `torch.testing.assert_close(out[0, 0], torch.tensor([[-1., 0.], [-1., -2.]]))`.

#### Assignment 8.2 — Automated Spatial Dimension Calculator
- **Objective**: Implement the exact spatial dimension formula for convolution and pooling layers.
- **Given**: Arbitrary combinations of $H_{in}, W_{in}$, kernel sizes, paddings, strides, and dilations.
- **Build / Do**:
  1. Implement a Python function `calc_conv2d_output_shape(in_shape, kernel_size, stride=1, padding=0, dilation=1)`.
  2. Test with 5 configurations including odd padding, dilation $> 1$, and non-square inputs.
  3. Validate against actual output shape of `nn.Conv2d`.
- **Pass Criteria**:
  - Calculated shapes match `nn.Conv2d(x).shape` across all 5 test cases.

#### Assignment 8.3 — Small CNN Classifier with Batch Normalization
- **Objective**: Build and train a modular CNN with convolution, batch normalization, pooling, and residual connections.
- **Given**: Synthetic image classification task with input shape `[B, 3, 32, 32]` and 10 classes.
- **Build / Do**:
  1. Construct CNN with 3 convolutional stages (channels: 16 $\to$ 32 $\to$ 64).
  2. Each stage contains: `Conv2d` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2)`.
  3. Replace fixed flattening with `nn.AdaptiveAvgPool2d((1, 1))` followed by `nn.Linear(64, 10)`.
  4. Train for 5 epochs on synthetic data.
- **Pass Criteria**:
  - Verify that model trains, parameters update, and loss decreases over 5 epochs.

#### Assignment 8.4 — Residual Block with Dimension-Matching Shortcut
- **Objective**: Implement a residual block that handles spatial downsampling and channel expansion.
- **Given**: Input feature map of shape `[B, 16, 32, 32]`, output channels $32$, stride $2$.
- **Build / Do**:
  1. Main path:
     - `Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)`
     - `BatchNorm2d(32)` $\to$ `ReLU`
     - `Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)`
     - `BatchNorm2d(32)`
  2. Shortcut path:
     - If stride $\ne 1$ or $C_{in} \ne C_{out}$: `nn.Sequential(Conv2d(16, 32, kernel_size=1, stride=2, bias=False), BatchNorm2d(32))`
     - Else: `nn.Identity()`
  3. Output: `F.relu(main_path(x) + shortcut_path(x))`.
- **Pass Criteria**:
  - Assert output shape is `(B, 32, 16, 16)`.
  - Test with stride 1 and matching channels; assert output shape equals input shape.

#### Assignment 8.5 — BatchNorm Train vs. Eval Activation Study
- **Objective**: Inspect the concrete numerical difference between BatchNorm in training mode vs. evaluation mode.
- **Given**: A small CNN containing `nn.BatchNorm2d`.
- **Build / Do**:
  1. In `model.train()` mode, pass a single batch of 4 images. Record output activations and running mean/variance.
  2. In `model.eval()` mode, pass the identical batch of 4 images. Record output activations.
  3. Pass a batch of size 1 in `eval()` mode and show that it succeeds using accumulated running stats.
  4. Pass a batch of size 1 in `train()` mode and observe the `ValueError` (cannot calculate variance for batch size 1).
- **Pass Criteria**:
  - Assert that activations differ between train and eval modes.
  - Successfully demonstrate that `eval()` allows single-sample inference while `train()` fails.

#### Assignment 8.6 — Resolution-Independent Feature Extraction with Hooks
- **Objective**: Use forward hooks to inspect feature activations from a CNN operating on variable image resolutions.
- **Given**: CNN using `nn.AdaptiveAvgPool2d((1, 1))` as the head.
- **Build / Do**:
  1. Register a forward hook on the final convolutional layer:
     ```python
     activations = []
     def hook_fn(module, input, output):
         activations.append(output.detach())
     hook = model.conv3.register_forward_hook(hook_fn)
     ```
  2. Pass an image batch of size `[2, 3, 32, 32]`. Record activation shape and output prediction shape.
  3. Pass an image batch of size `[2, 3, 64, 64]`. Record activation shape and output prediction shape.
  4. Clean up: `hook.remove()`.
- **Pass Criteria**:
  - Assert that 32x32 input produces intermediate activation `[2, 64, 8, 8]` and output `[2, 10]`.
  - Assert that 64x64 input produces intermediate activation `[2, 64, 16, 16]` and output `[2, 10]`.
  - Verify that the final classification head works seamlessly across both resolutions.

**Exit criterion:** You can design a CNN, calculate spatial dimensions through convolutions and pooling, use residual connections, and inspect feature activations with hooks.

---

## Chapter 9 — Embeddings, sequences, masking, and attention

### What are we learning in this chapter?
- **Token Embeddings from Scratch**: Why `nn.Embedding` is simply a fast lookup table, and why it is mathematically identical to multiplying a one-hot vector by a weight matrix.
- **Attention from First Principles**: The Query, Key, Value analogy (searching, matching, and retrieving information).
- **Scaled Dot-Product Attention**: Why dividing by $\sqrt{d_k}$ is mathematically necessary to prevent vanishing gradients in softmax.
- **Multi-Head Attention (MHA)**: How projecting into multiple subspaces allows the model to attend to different types of relationships simultaneously.
- **Masking Mechanisms**: Why causal masks prevent tokens from peeking into the future, and how padding masks prevent attention on blank tokens.
- **Quadratic Complexity**: Understanding why attention memory scales as $\mathcal{O}(T^2)$ with respect to sequence length.

---

### 1. Embeddings: Words as Vectors

Computers cannot process words like `"cat"` or `"bank"` directly. They need numbers.
We give every word in our vocabulary an integer ID:
`"the" -> 0`, `"cat" -> 1`, `"sat" -> 2`.

An **Embedding** (`nn.Embedding(vocab_size, embedding_dim)`) is simply a table of learnable vectors.
If `vocab_size = 10` and `embedding_dim = 4`, it is a $10 \times 4$ weight matrix.
When you pass word ID `1`, PyTorch grabs row 1:
```python
embedding = nn.Embedding(10, 4)
word_vector = embedding(torch.tensor([1]))  # Grabs row 1: [4 numbers]
```

**The One-Hot Equivalence:**
Grabbing row 1 is mathematically identical to creating a one-hot vector `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` and multiplying it by the embedding matrix!
`nn.Embedding` is just an optimized shortcut that skips the matrix multiplication.

---

### 2. Attention: The Query-Key-Value Analogy

Imagine you are looking for a video tutorial on YouTube:
1. **Query ($Q$):** What *you* are searching for (e.g. `"how to bake bread"`).
2. **Key ($K$):** The *titles and tags* of every video in YouTube's database.
3. **Value ($V$):** The *actual video content* you watch.

In self-attention, every word in a sentence generates its own Query, Key, and Value:
1. Compare my Query with every word's Key (using dot-product: $Q \cdot K^T$). This gives a **compatibility score**.
2. Pass the scores through `softmax` so they become percentages that sum to $100\%$ (attention weights).
3. Compute a weighted average of all the Values based on those percentages!

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V$$

#### Why divide by $\sqrt{d_k}$?
If the head dimension $d_k$ is large (say, 64 or 128), dot-products can grow to large numbers like $+80$ or $-80$.
When you feed large numbers into `softmax`, the output becomes extremely peaked: one value becomes `1.0` and all others become `0.0`.
In those extreme regions, the derivative of softmax is virtually **zero**! Gradients vanish and the network stops learning.
Dividing by $\sqrt{d_k}$ keeps the variance of the scores at $1.0$, keeping softmax in its active, learnable sweet spot.

---

### 3. Masking: Causal vs. Padding

- **Causal Mask (Autoregressive):**
  When generating text, word 3 is not allowed to see word 4 or word 5!
  We enforce this by creating a lower-triangular boolean mask with `torch.tril` and filling future positions with $-\infty$ using `.masked_fill()`:
  ```python
  T = 4
  # 1. Lower triangular boolean mask (True on and below diagonal)
  causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
  
  # 2. Additive float mask (-inf above diagonal)
  float_mask = torch.zeros(T, T).masked_fill(~causal_mask, float("-inf"))
  # [[ 0.0, -inf, -inf, -inf],
  #  [ 0.0,  0.0, -inf, -inf],
  #  [ 0.0,  0.0,  0.0, -inf],
  #  [ 0.0,  0.0,  0.0,  0.0]]
  
  # When added to scores before softmax, e^(-inf) evaluates to exactly 0.0!
  attn_weights = F.softmax(scores + float_mask, dim=-1)
  ```

- **Padding Mask:**
  If a sentence was padded with zeros to fit a batch, we mask out the padding tokens so real words don't waste attention on meaningless filler.

---

### 4. Multi-Head Projections & `F.scaled_dot_product_attention`

Rather than computing attention once across the full dimension $D$, **Multi-Head Attention (MHA)** splits $D$ into $H$ independent heads, each with dimension $D_h = D / H$:
1. Input $x$ has shape `[B, T, D]`.
2. Project $x$ through linear layers to get $Q, K, V$ of shape `[B, T, D]`.
3. Reshape and permute into `[B, H, T, Dh]`:
   ```python
   B, T, D = x.shape
   H = 4
   Dh = D // H
   # [B, T, D] -> [B, T, H, Dh] -> [B, H, T, Dh]
   Q = W_q(x).view(B, T, H, Dh).transpose(1, 2)
   K = W_k(x).view(B, T, H, Dh).transpose(1, 2)
   V = W_v(x).view(B, T, H, Dh).transpose(1, 2)
   ```
4. Compute attention:
   ```python
   # Modern PyTorch provides native FlashAttention-accelerated kernel:
   out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
   # Recombine heads: [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
   out = out.transpose(1, 2).contiguous().view(B, T, D)
   ```

---

### Assignments 9

#### Assignment 9.1 — Embedding Lookup vs. Matrix Multiplication
- **Objective**: Prove that `nn.Embedding` is mathematically equivalent to multiplying a one-hot vector by an embedding matrix.
- **Given**: Vocabulary size $V = 10$, embedding dimension $D = 4$, token IDs `tokens = torch.tensor([2, 5, 0], dtype=torch.long)`.
- **Build / Do**:
  1. Instantiate `embedding = nn.Embedding(10, 4)`.
  2. Extract embeddings directly: `out_embed = embedding(tokens)`.
  3. Construct one-hot vectors: `one_hot = F.one_hot(tokens, num_classes=10).float()`.
  4. Multiply one-hot matrix by embedding weights: `out_matmul = one_hot @ embedding.weight`.
- **Pass Criteria**:
  - Assert `torch.testing.assert_close(out_embed, out_matmul)`.

#### Assignment 9.2 — Scaled Dot-Product Attention by Hand & Verification
- **Objective**: Implement scaled dot-product attention from raw tensor operations and verify against `F.scaled_dot_product_attention`.
- **Given**:
  - Query, Key, Value tensors of shape `[B, H, T, Dh] = [1, 1, 3, 4]`.
  - Manual numerical values for $Q, K, V$.
- **Build / Do**:
  1. Compute attention scores: `scores = (Q @ K.transpose(-2, -1)) / math.sqrt(4)`.
  2. Compute attention probabilities: `attn_weights = F.softmax(scores, dim=-1)`.
  3. Compute weighted values: `out_manual = attn_weights @ V`.
  4. Compute with PyTorch native: `out_native = F.scaled_dot_product_attention(Q, K, V)`.
- **Pass Criteria**:
  - Assert `torch.testing.assert_close(out_manual, out_native)`.
  - Verify that each row of `attn_weights` sums to $1.0$.

#### Assignment 9.3 — Constructing a Causal Attention Mask
- **Objective**: Construct an upper-triangular causal attention mask that prevents information flow from future tokens.
- **Given**: Sequence length $T = 5$.
- **Build / Do**:
  1. Construct boolean lower-triangular mask: `causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))`.
  2. Construct additive float mask: `float_mask = torch.zeros(5, 5).masked_fill(~causal_mask, float("-inf"))`.
  3. Apply `float_mask` to random scores of shape `[5, 5]` and compute softmax.
- **Pass Criteria**:
  - Assert `causal_mask[i, j] == True` for $j \le i$ and `False` for $j > i$.
  - Assert that in `F.softmax(scores + float_mask, dim=-1)`, all upper-triangular positions ($j > i$) evaluate to strictly $0.0$.

#### Assignment 9.4 — Multi-Head Attention Module from Scratch
- **Objective**: Implement a modular `MultiHeadAttention` class using linear projections and reshape/permute operations.
- **Given**: Model dimension $D = 64$, number of heads $H = 4$ (head dimension $D_h = 16$).
- **Build / Do**:
  1. Define `MyMultiHeadAttention(d_model=64, n_heads=4)`:
     - Linear layers: `q_proj`, `k_proj`, `v_proj`, `out_proj`.
     - In `forward(x, mask=None)`:
       - Project $Q, K, V$: `[B, T, 64]`.
       - Reshape and permute to `[B, 4, T, 16]`.
       - Compute attention using `F.scaled_dot_product_attention(q, k, v, attn_mask=mask)`.
       - Permute and reshape back to `[B, T, 64]`.
       - Apply `out_proj`.
  2. Test with input `x = torch.randn(2, 10, 64)`.
- **Pass Criteria**:
  - Assert output shape is `(2, 10, 64)`.
  - Test with causal mask; verify output is finite and shapes match.

#### Assignment 9.5 — Combined Causal and Padding Mask
- **Objective**: Combine an autoregressive causal mask with variable-length padding masks.
- **Given**: Batch of 2 sequences:
  - Sequence 1: 4 valid tokens (length 4)
  - Sequence 2: 2 valid tokens, 2 padding tokens (length 2, padded to 4)
- **Build / Do**:
  1. Construct causal mask of shape `[4, 4]`.
  2. Construct padding mask of shape `[2, 1, 1, 4]` (`True` for valid tokens, `False` for padding).
  3. Combine masks: `combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask`.
  4. Apply combined mask to dummy attention scores and compute softmax.
- **Pass Criteria**:
  - Assert that for sequence 2, positions corresponding to padding tokens have attention weight $0.0$ at all timesteps.
  - Assert that for sequence 2, at timestep 0, attention weight is $1.0$ on token 0 and $0.0$ on tokens 1, 2, 3.

#### Assignment 9.6 — Empirical Attention Complexity Scaling
- **Objective**: Measure and verify the $\mathcal{O}(T^2)$ time and memory scaling of self-attention.
- **Given**: Sequence lengths $T \in \{128, 256, 512, 1024, 2048\}$, fixed $D = 64$.
- **Build / Do**:
  1. For each sequence length, allocate `Q, K, V = torch.randn(1, 4, T, 16)`.
  2. Measure execution time of `F.scaled_dot_product_attention` over 50 iterations after warm-up.
  3. Compute ratio of time between $T$ and $2T$.
- **Pass Criteria**:
  - Tabulate sequence length vs. execution time.
  - Show that doubling sequence length results in approximately $4\times$ increase in attention matrix computation time as $T$ grows.

**Exit criterion:** You can implement scaled dot-product attention, explain the $\sqrt{d_k}$ factor, build multi-head projections, and construct causal/padding masks.

---

## Chapter 10 — A transformer from first principles

### What are we learning in this chapter?
- **The Decoder-Only Transformer Architecture**: Assembling a modern autoregressive transformer (like GPT) from fundamental blocks.
- **The Pre-LN Transformer Block**: Why placing LayerNorm before attention and MLP stabilizes training in deep networks.
- **Next-Token Prediction Targets**: How to correctly shift inputs and targets (`tokens[:, :-1]` vs. `tokens[:, 1:]`).
- **Weight Tying**: Reusing the input embedding matrix as the final linear projection head to reduce parameters and improve generalization.
- **Sampling from Next-Token Logits**: Implementing greedy decoding, temperature scaling, and top-k filtering.
- **Key-Value (KV) Caching**: How to avoid recomputing past tokens during generation, slashing inference time from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$.

---

### 1. The Anatomy of a Transformer Block

In modern language models, we stack identical **Transformer Blocks**.
The standard modern architecture uses **Pre-LN** (LayerNorm before each sub-layer):

```
Input x
  |---------------------------------\ (Residual Highway)
  v                                 |
LayerNorm                           |
  v                                 |
Causal Multi-Head Attention         |
  v                                 |
  + <-------------------------------/
  |
  |---------------------------------\ (Residual Highway)
  v                                 |
LayerNorm                           |
  v                                 |
MLP: Linear -> GELU -> Linear       |
  v                                 |
  + <-------------------------------/
  |
Output
```

---

### 2. Next-Token Prediction: Shifting Inputs and Targets

Language modeling is simple: given all past words, predict the very next word!
If our sequence is: `[The, cat, sat, on, the, mat]`:
- **Input ($x$):** `[The, cat, sat, on, the]` (all tokens except the last)
- **Target ($y$):** `[cat, sat, on, the, mat]` (all tokens except the first)

```python
x = tokens[:, :-1]  # Shape: [B, T-1]
y = tokens[:, 1:]   # Shape: [B, T-1]
```
When token 0 (`"The"`) goes through the model, its output logit is compared against target 0 (`"cat"`).

---

### 3. Text Generation: Temperature and Top-K

When generating new text:
1. Pass prompt tokens into the model to get logits for the next token: `logits` of shape `[1, VocabSize]`.
2. **Temperature ($\tau$):** Divide logits by temperature:
   $$\text{logits} = \frac{\text{logits}}{\tau}$$
   - High temperature (e.g. 1.5): Flattens probabilities $\implies$ more creative / random text.
   - Low temperature (e.g. 0.2): Sharpens probabilities $\implies$ more predictable / repetitive text.
   - Near zero ($0.001$): Equivalent to `argmax` (greedy choice).
3. **Top-K Filtering:** Only keep the top $K$ highest logits; set all others to $-\infty$:
   ```python
   # Filter logits to top-k:
   v, _ = torch.topk(logits, k=5)
   logits[logits < v[:, [-1]]] = float("-inf")
   probs = F.softmax(logits, dim=-1)
   next_token = torch.multinomial(probs, num_samples=1)
   ```

---

### 4. KV-Caching: Fast Autoregressive Generation

In naive generation, to generate token 100, you pass all 99 previous tokens through the model. To generate token 101, you pass all 100 tokens.
This means you are recomputing the Key and Value vectors of past tokens over and over!
With **KV-Caching**:
- Compute and save $K$ and $V$ for each token once.
- When generating the next token, pass **only the single new token** through the model!
- Concatenate its new Key and Value along the sequence dimension (`dim=-2`):
  ```python
  if past_k is not None:
      K = torch.cat([past_k, K], dim=-2)
      V = torch.cat([past_v, V], dim=-2)
  # Update cache for next step:
  cache = (K, V)
  ```
This reduces generation time from quadratic $\mathcal{O}(T^2)$ to linear $\mathcal{O}(T)$!

---

### 5. Positional Embeddings & Weight Tying

#### A. Positional Embeddings
Because attention is permutation-invariant (it treats all words as an unordered set), the model cannot distinguish `"dog bites man"` from `"man bites dog"`.
We must inject position information by adding a **positional embedding**:
```python
token_embed = nn.Embedding(vocab_size, d_model)
pos_embed = nn.Embedding(max_seq_len, d_model)

# Forward pass:
B, T = tokens.shape
pos_ids = torch.arange(T, device=tokens.device)  # [0, 1, ..., T-1]
x = token_embed(tokens) + pos_embed(pos_ids)     # Shape: [B, T, d_model]
```

#### B. Weight Tying
In language models, the final linear head projects $D \to V$ (`nn.Linear(d_model, vocab_size, bias=False)`).
Instead of allocating a separate $V \times D$ matrix, we can **tie the weights** of the head directly to the input token embedding matrix:
```python
lm_head = nn.Linear(d_model, vocab_size, bias=False)
# Point the linear weight directly to the embedding weight:
lm_head.weight = token_embed.weight
```
This saves millions of parameters and forces the input and output representations to align in the same geometric space!

---

### Assignments 10

#### Assignment 10.1 — Complete Tiny Decoder-Only Transformer
- **Objective**: Assemble a complete, functional decoder-only autoregressive transformer model.
- **Given**:
  - Vocab size $V = 100$, maximum sequence length $T = 32$, $D = 64$, heads $H = 4$, layers $L = 2$.
- **Build / Do**:
  1. Build `TransformerBlock`: Pre-LN `LayerNorm`, `MyMultiHeadAttention`, residual connection, `LayerNorm`, MLP (`Linear(64, 256) -> GELU -> Linear(256, 64)`), residual connection.
  2. Build `DecoderLM`:
     - Token embedding `nn.Embedding(V, D)` + Learnable positional embedding `nn.Embedding(T, D)`.
     - Stack of 2 `TransformerBlock` modules.
     - Final `LayerNorm(D)` and classification head `nn.Linear(D, V, bias=False)`.
  3. Test forward pass with random token sequence `tokens = torch.randint(0, 100, (2, 16))`.
- **Pass Criteria**:
  - Assert output logits shape is exactly `(2, 16, 100)`.
  - Assert model runs forward and backward passes without error.

#### Assignment 10.2 — Overfitting a Single Sequence to Near-Zero Loss
- **Objective**: Prove the transformer model implementation has sufficient capacity by memorizing a small sequence.
- **Given**: A fixed sequence of 16 tokens: `tokens = torch.tensor([[1, 5, 12, 8, 9, 42, 11, 7, 3, 19, 22, 31, 14, 2, 8, 10]])`.
- **Build / Do**:
  1. Set inputs: `x = tokens[:, :-1]`, targets: `y = tokens[:, 1:]`.
  2. Train `DecoderLM` using `AdamW(lr=1e-3)` and `nn.CrossEntropyLoss()`.
  3. Train for 150 steps.
- **Pass Criteria**:
  - Training loss drops below `0.05`.
  - Greedy generation from the prompt `tokens[:, :3]` reproduces the exact remainder of the sequence.

#### Assignment 10.3 — Temperature and Top-K Generation Controller
- **Objective**: Implement text generation with adjustable temperature and top-k filtering.
- **Given**: Logits vector for next token: `logits = torch.randn(1, 100)`.
- **Build / Do**:
  1. Implement `generate_next_token(logits, temperature=1.0, top_k=None)`:
     - Scale by temperature: `logits = logits / max(temperature, 1e-5)`.
     - If `top_k` is specified: find the $k$-th largest value using `torch.topk`; set all logits smaller than this threshold to `-float("inf")`.
     - Compute probabilities with `F.softmax(logits, dim=-1)`.
     - Sample token index using `torch.multinomial(probs, num_samples=1)`.
  2. Test with `temperature=0.01` (should behave like `argmax`).
  3. Test with `top_k=5` (verify sampled index is always in top 5).
- **Pass Criteria**:
  - Assert that near-zero temperature ($0.001$) matches `logits.argmax(dim=-1)` on 20 trials.
  - Assert that with `top_k=5`, 100 random draws never produce an index outside the top 5 largest logits.

#### Assignment 10.4 — Weight Tying Implementation and Parameter Count
- **Objective**: Tie the weights of the input embedding and output projection layer and verify parameter reduction.
- **Given**: `DecoderLM` with $V = 1,000$ and $D = 128$.
- **Build / Do**:
  1. Count total parameters before weight tying.
  2. Tie weights: `model.head.weight = model.token_embedding.weight`.
  3. Count total parameters after weight tying.
  4. Perform one training step and verify gradients accumulate into the shared weight tensor.
- **Pass Criteria**:
  - Assert total parameters decrease by exactly $V \times D = 1,000 \times 128 = 128,000$ elements.
  - Assert `model.head.weight is model.token_embedding.weight`.
  - Assert `model.token_embedding.weight.grad` receives gradients from both embedding lookup and output projection.

#### Assignment 10.5 — Padding-Safe Language Modeling Loss
- **Objective**: Compute cross-entropy loss over padded sequences while completely ignoring padding tokens.
- **Given**:
  - Logits: shape `[2, 4, 10]`
  - Targets: shape `[2, 4]` with padding token ID `0`:
    - Row 1: `[5, 8, 2, 0]` (last token is pad)
    - Row 2: `[3, 4, 0, 0]` (last two tokens are pad)
- **Build / Do**:
  1. Flatten logits to `[8, 10]` and targets to `[8]`.
  2. Compute loss with `nn.CrossEntropyLoss(ignore_index=0)`.
  3. Manually compute cross-entropy on only the non-padded tokens (positions `(0,0), (0,1), (0,2), (1,0), (1,1)`).
- **Pass Criteria**:
  - Assert that `nn.CrossEntropyLoss(ignore_index=0)` matches the manual non-padded average exactly.

#### Assignment 10.6 — Key-Value (KV) Cache for Inference Acceleration
- **Objective**: Implement a KV-cache to avoid recomputing past key and value representations during autoregressive generation.
- **Given**: A 1-layer transformer block.
- **Build / Do**:
  1. Standard generation: At step $t$, pass all tokens $0 \dots t$ through the transformer block to get next-token logits.
  2. Cached generation:
     - At step $0$, compute and store $K_0, V_0$.
     - At step $t > 0$, pass only the single new token $x_t$ through $Q, K, V$ projections.
     - Concatenate new $K_t, V_t$ with cached past keys and values: `K = torch.cat([K_cache, K_new], dim=-2)`.
     - Compute attention using $Q_t$ against the full concatenated $K$ and $V$.
- **Pass Criteria**:
  - Assert that output logits from cached generation match standard full-sequence generation within `1e-5`.
  - Verify that query tensor has sequence length 1 during cached steps ($t > 0$).

#### Assignment 10.7 — Transformer Architecture Closed-Book Retrieval
- **Objective**: Reconstruct the complete transformer forward pass from memory without reference materials.
- **Given**: An empty file.
- **Build / Do**:
  - Write from memory: Attention scaling equation, multi-head projection shapes, Pre-LN block architecture, and autoregressive causal masking.
- **Pass Criteria**:
  - Clean script runs forward pass, computes loss, and performs backward pass without error.

**Exit criterion:** You can implement a complete decoder-only transformer, explain Pre-LN stability, tie embedding weights, and generate text with temperature and top-k controls.

---

# Part IV — Engineering and Scale

## Chapter 11 — Transfer learning and fine-tuning

### What are we learning in this chapter?
- **Why Fine-Tuning Works**: Standing on the shoulders of giants by adapting large pretrained backbones rather than training from scratch.
- **Freezing Mechanics & Parameter Audits**: Toggling `param.requires_grad = False` and verifying which weights update.
- **Discriminative Learning Rates**: Assigning different learning rates to different layers via optimizer parameter groups.
- **Parameter-Efficient Fine-Tuning (PEFT) & LoRA**: The Low-Rank Adaptation formula ($W_0 + \frac{\alpha}{r} BA$) and why initializing $B=0$ guarantees that step 0 behaves identically to the pretrained model.

---

### 1. What problem does transfer learning solve?

Training a vision model (like a ResNet or ViT) from scratch requires millions of images and weeks of GPU compute.
Training a modern language model from scratch requires millions of dollars in compute.
Worse, if your personal dataset has only 1,000 samples, training from scratch will immediately overfit and fail.

**Transfer Learning:**
Pretrained models have already learned universal representations:
- Early layers in vision models detect edges, textures, and geometric shapes.
- Early layers in language models understand grammar, syntax, and sentence structure.

Instead of retraining everything, we take the pretrained model, throw away its final classification head, attach a new head tailored to our task, and train!

---

### 2. The Two-Stage Fine-Tuning Strategy

1. **Stage 1: Head Warmup (Freezing the Backbone)**
   - Freeze all pretrained layers: `for p in model.backbone.parameters(): p.requires_grad = False`.
   - Train *only* your new classification head for 3–5 epochs.
   - *Why?* At the start, the new head has random weights. If you backpropagate through the whole model immediately, the large, chaotic gradients from the untrained head will destroy ("catastrophically forget") the rich features learned by the backbone.
2. **Stage 2: End-to-End Fine-Tuning (Unfreezing with Low LR)**
   - Unfreeze the backbone: `p.requires_grad = True`.
   - Train the whole model with a tiny learning rate (e.g. $10^{-5}$ for backbone, $10^{-4}$ for head).

#### Auditing Parameter Freeze Status with `named_parameters()`
Before training, always verify which layers are trainable and which are frozen using `model.named_parameters()`:
```python
def audit_parameters(model):
    total, trainable, frozen = 0, 0, 0
    print(f"{'Parameter Name':<35} {'Shape':<15} {'Requires Grad'}")
    print("-" * 65)
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
        print(f"{name:<35} {str(list(param.shape)):<15} {param.requires_grad}")
    print("-" * 65)
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {frozen:,}")
```
This guarantees you never accidentally update a layer you intended to freeze or leave your new head frozen.

---

### 3. Low-Rank Adaptation (LoRA) from First Principles

In models with billions of parameters, even saving fine-tuned checkpoints is expensive.
**LoRA (Low-Rank Adaptation)** freezes the original weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and decomposes the update matrix $\Delta W$ into two tiny matrices:
$$\Delta W = \frac{\alpha}{r} (B \times A)$$
Where:
- $W_0$ is frozen ($d \times k$).
- $A \in \mathbb{R}^{r \times k}$ is initialized with small random Gaussian values.
- $B \in \mathbb{R}^{d \times r}$ is initialized to **all zeros**!
- $r$ is the rank (typically $4$ or $8$), which is vastly smaller than $d$ (e.g. 4096).

```
          Original Frozen Weight W0 (d x k)
                   [4096 x 4096]  (16.7M weights)
                         +
   Adapter B (d x r)            Adapter A (r x k)
     [4096 x 8]         x          [8 x 4096]
   (32k weights)                 (32k weights)
Total trainable weights: 65,536 (99.6% reduction!)
```

**Why initialize $B$ to zero?**
Because at step 0, $B \times A = 0 \times A = 0$!
The initial output of the model is 100% identical to the original pretrained model. There is zero disruption at the start of training!

---

### 4. Discriminative Learning Rates via Parameter Groups

Different parts of a fine-tuned network should learn at different speeds:
- **The Pretrained Backbone:** Should be adjusted gently with a small learning rate (e.g. $10^{-5}$) so pretrained knowledge is not destroyed.
- **The New Head:** Should learn quickly with a larger learning rate (e.g. $10^{-3}$) to adapt to the new class labels.

Pass a list of parameter dictionaries into your optimizer:
```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(),     "lr": 1e-3},
], weight_decay=1e-2)
```

---

### 5. Implementing a Custom `LoRALinear` Layer

Here is how LoRA is implemented as an `nn.Module` wrapper around an existing linear layer:
```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False
            
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.scaling = alpha / rank
        
        # LoRA matrices:
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))  # Initialized to ZERO!
        
    def forward(self, x):
        # Base forward + scaled low-rank update
        # x is [B, in_features]
        # x @ A.t() is [B, rank]
        # (x @ A.t()) @ B.t() is [B, out_features]
        lora_out = (x @ self.A.t()) @ self.B.t()
        return self.base_layer(x) + self.scaling * lora_out
```

---

### Assignments 11

#### Assignment 11.1 — Two-Stage Fine-Tuning Protocol
- **Objective**: Implement a two-stage transfer learning pipeline: head warmup followed by backbone fine-tuning.
- **Given**: A pretrained model (or synthetic feature extractor + linear head).
- **Build / Do**:
  1. Stage 1 (Head Warmup):
     - Freeze backbone: for `param in model.backbone.parameters(): param.requires_grad = False`.
     - Create optimizer with only head parameters: `optimizer = AdamW(model.head.parameters(), lr=1e-3)`.
     - Train for 3 epochs.
  2. Stage 2 (End-to-End Fine-Tuning):
     - Unfreeze backbone: for `param in model.backbone.parameters(): param.requires_grad = True`.
     - Create optimizer with discriminative learning rates:
       `optimizer = AdamW([{"params": model.backbone.parameters(), "lr": 1e-5}, {"params": model.head.parameters(), "lr": 1e-4}])`.
     - Train for 3 epochs.
- **Pass Criteria**:
  - In Stage 1: assert that backbone parameter gradients are `None` and backbone weights remain identical to initialization.
  - In Stage 2: assert that both backbone and head parameter gradients exist and weights update.

#### Assignment 11.2 — Parameter Freeze Audit Tool
- **Objective**: Build an audit function that verifies parameter freeze status and reports memory allocation.
- **Given**: A model with partially frozen layers.
- **Build / Do**:
  1. Write `audit_parameters(model)`:
     - Iterate through `model.named_parameters()`.
     - Print/tabulate: parameter name, shape, `requires_grad` status, and element count.
     - Calculate total parameters, trainable parameters, and frozen parameters.
  2. Test on a model where early layers are frozen and head is unfrozen.
- **Pass Criteria**:
  - Audit report accurately identifies frozen vs. trainable layers.
  - Assert that `total_params == trainable_params + frozen_params`.

#### Assignment 11.3 — Low-Rank Adaptation (LoRA) Linear Layer from Scratch
- **Objective**: Implement a LoRA linear layer and verify that it preserves the base layer's exact outputs at initialization.
- **Given**: Base linear layer `base = nn.Linear(32, 64)`, rank $r = 4$, scaling factor $\alpha = 8.0$.
- **Build / Do**:
  1. Define `LoRALinear(base_layer, rank=4, alpha=8.0)`:
     - Freeze `base_layer.weight` and `base_layer.bias`.
     - Register `self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)`.
     - Register `self.lora_B = nn.Parameter(torch.zeros(out_features, rank))`.
     - In `forward(x)`:
       $$\text{output} = \text{base\_layer}(x) + \frac{\alpha}{r} (x @ A^T @ B^T)$$
  2. Pass input `x = torch.randn(4, 32)` through `base_layer` and `lora_layer` at step 0.
- **Pass Criteria**:
  - At step 0: assert `torch.testing.assert_close(lora_layer(x), base_layer(x))` (exact output equality because $B = 0$).
  - After one training step: assert that `lora_A` and `lora_B` receive gradients while `base_layer.weight.grad` is `None`.

#### Assignment 11.4 — Parameter Groups with Custom Weight Decay
- **Objective**: Configure an optimizer with separate parameter groups for weights (with decay) and biases/LayerNorms (without decay).
- **Given**: A model with multiple linear and normalization layers.
- **Build / Do**:
  1. Split parameters into two groups:
     - Decay group: 2D weight matrices (e.g. `p.ndim >= 2`) with `weight_decay=1e-2`.
     - No-decay group: 1D biases and normalization scales/shifts (`p.ndim < 2`) with `weight_decay=0.0`.
  2. Instantiate `AdamW(param_groups, lr=1e-3)`.
- **Pass Criteria**:
  - Assert that every parameter in `model.parameters()` belongs to exactly one group (no omissions, no duplicates).
  - Assert that bias and normalization parameters have `weight_decay == 0.0`.

**Exit criterion:** You can freeze/unfreeze model backbones, configure discriminative parameter groups, and implement LoRA adapters from scratch.

---

## Chapter 12 — Debugging numerical and training failures

### What are we learning in this chapter?
- **The 5-Step Debugging Ladder**: A systematic protocol for diagnosing training bugs instead of randomly tweaking hyperparameters.
- **Forward Hooks for Activation Inspection**: Catching exploding activations, vanishing values, and `NaN`/`Inf` at the exact layer where they originate.
- **Retained-Graph Memory Leaks**: Understanding why appending raw `loss` tensors to a Python list causes silent Out-of-Memory (OOM) crashes.
- **Numerical Gradient Verification**: Using `torch.autograd.gradcheck` to scientifically validate custom backward formulas.
- **The Permutation Test (Shuffled Labels)**: How to prove whether poor accuracy is caused by an architecture bug or corrupted training labels.
- **Contextual Exception Handling**: Wrapping dataset loaders to report the exact failing sample ID and file path.

---

### 1. The 5-Step Debugging Ladder

When a model fails to train or outputs `NaN`, amateurs immediately start changing learning rates or adding layers.
Master engineers follow this rigorous 5-step ladder:

```
[ Step 1: DATA ] ----------> Check shapes, dtypes, ranges [min, max],
                             verify no NaNs/Infs, check class balance.
       |
[ Step 2: BOUNDARY ] ------> Assert model output shapes match target shapes.
                             Check loss contract (e.g. logits, not probs).
       |
[ Step 3: LOSS ] ----------> Check initial loss value. For C classes,
                             initial CrossEntropyLoss must be ~ -ln(1/C).
       |
[ Step 4: GRADIENTS ] -----> Inspect parameter .grad: Are they None? Zero?
                             Exploding (> 100)? Or NaN?
       |
[ Step 5: UPDATES ] -------> Clone weights before and after optimizer.step().
                             Did the weights actually move?
```

---

### 2. Forward Hooks: Catching NaNs in the Act

If your model output becomes `NaN`, where did the bug happen? Layer 1? Layer 20? Layer 50?
Using **Forward Hooks**, you can attach an inspector to every single layer:

```python
def make_inspector(name):
    def hook(module, input, output):
        # Detach to avoid holding computation graphs in memory!
        out = output.detach()
        if not torch.isfinite(out).all():
            print(f"CRITICAL: Non-finite values detected in layer: {name}!")
            print(f"Shape: {out.shape} | NaNs: {torch.isnan(out).sum()} | Infs: {torch.isinf(out).sum()}")
    return hook

# Register hooks on all linear layers
handles = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        handles.append(module.register_forward_hook(make_inspector(name)))

# Clean up hooks when done:
# for h in handles: h.remove()
```

---

### 3. The Retained-Graph Memory Leak

One of the most frequent OOM crashes in PyTorch happens with loss logging:
```python
# CATASTROPHIC MEMORY LEAK!
history = []
for batch in dataloader:
    loss = criterion(model(x), y)
    history.append(loss)  # WRONG! Keeps the ENTIRE computation graph alive in RAM!
```
Because `loss` has a `.grad_fn`, it holds references to every activation and weight in the entire network for that batch! Storing 1,000 `loss` tensors means storing 1,000 full computation graphs in memory!

**The Fix:**
Always store the raw Python float:
```python
history.append(loss.item())  # Discards the graph, stores only a scalar number!
```

**Measuring the Leak (`torch.cuda.memory_allocated()`):**
You can directly observe this memory bloat on GPU using `torch.cuda.memory_allocated()`, which returns the exact number of bytes occupied by live PyTorch tensors:
```python
# Leaky run: memory increases linearly with every iteration!
# print(f"GPU Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
# Correct run (.item()): memory stays completely flat across all iterations.
```

---

### 4. Catching Backward NaNs: `torch.autograd.detect_anomaly()`

Sometimes, a forward pass finishes without error, but calling `loss.backward()` crashes with:
`RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.`
Where did this `nan` come from?
Standard backpropagation cannot tell you because it only knows the backward stack.
Enable **Anomaly Detection**:
```python
# Wrap your training step in the anomaly detection context:
with torch.autograd.detect_anomaly():
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
```
When a NaN occurs, PyTorch prints the exact traceback of the **forward operation** that created the offending value (e.g. line 42: `torch.log(prob)`), making the bug instantly solvable!

---

### 5. Gradient Exploding & Clipping: `clip_grad_norm_`

In deep models, gradients can multiply across layers and explode to huge values like $10^5$, causing weights to take massive steps into unstable territory and destroy the network.
**Gradient Clipping** scales all parameter gradients down if their combined total $L_2$ norm exceeds a threshold:
$$\mathbf{g} \leftarrow \mathbf{g} \times \frac{\text{max\_norm}}{\max(\|\mathbf{g}\|_2, \text{max\_norm})}$$

Call it immediately before `optimizer.step()`:
```python
loss.backward()

# Clip total gradient norm to 1.0:
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

---

### 6. Scientifically Validating Custom Layers: `torch.autograd.gradcheck`

When writing a custom `torch.autograd.Function`, how do you know your `backward()` method is mathematically exact?
PyTorch provides **`torch.autograd.gradcheck`**, which compares your analytical backward gradients against numerical finite-difference approximations:
```python
from torch.autograd import gradcheck

# Must use float64 (double precision) to prevent numerical cancellation:
inputs = (torch.randn(3, 3, dtype=torch.float64, requires_grad=True),)

# gradcheck returns True if analytical gradients match numerical gradients:
test_passed = gradcheck(my_custom_function, inputs, eps=1e-6, atol=1e-4)
assert test_passed
```

---

### Assignments 12

#### Assignment 12.1 — Bug Zoo: Diagnosis and Repair
- **Objective**: Diagnose and fix 5 distinct common bugs in single-batch training scripts.
- **Given**: Five broken scripts:
  1. Target dtype mismatch: `torch.float32` passed to `nn.CrossEntropyLoss` expecting `torch.long`.
  2. Silent broadcasting bug: target `[B]` subtracted from prediction `[B, 1]` in MSE loss.
  3. Detached logits: `logits = head(features).detach()` breaking backprop.
  4. Missing head parameters: replacing model head without updating optimizer parameter list.
  5. Negative log argument: evaluating `torch.log(x)` where $x \le 0$ producing `NaN`.
- **Build / Do**:
  1. Run each script and record the exact failure symptom (exception or silent failure).
  2. Apply the minimal repair without using broad `try...except` blocks.
- **Pass Criteria**:
  - Write a bug report for all 5 cases detailing: Visible Symptom, Confirming Diagnostic, Root Cause, and Minimal Fix.

#### Assignment 12.2 — Activation Inspector Hook Suite
- **Objective**: Implement automated forward hooks to detect exploding/vanishing activations and non-finite values.
- **Given**: A multi-layer neural network.
- **Build / Do**:
  1. Write an `ActivationInspector` class:
     - Registers forward hooks on all leaf layers (`nn.Linear`, `nn.Conv2d`).
     - For each activation, records: layer name, shape, dtype, mean, standard deviation, min, max, and count of `NaN`/`Inf` elements.
     - Safely unregisters all hooks in a `finally` block or via context manager.
     - Ensures detached tensors are stored to avoid memory leaks.
  2. Run with a valid batch, then with an input containing a single `NaN`.
- **Pass Criteria**:
  - Correctly logs activation statistics across all layers.
  - Successfully catches and pinpoints the first layer where `NaN` appears.

#### Assignment 12.3 — Retained-Graph Memory Leak Experiment
- **Objective**: Demonstrate and measure the memory leak caused by storing raw loss tensors in a history list.
- **Given**: A small MLP and fixed input batch.
- **Build / Do**:
  1. Leaky run: Run 20 forward passes (without backward), appending `loss` directly to `history = []`.
  2. Correct run: Run 20 forward passes, appending `loss.item()` to `history = []`.
  3. In both runs, measure memory allocation using `torch.cuda.memory_allocated()` (or process RSS if CPU).
- **Pass Criteria**:
  - Show that the leaky run retains computation graphs, causing linear memory growth.
  - Show that the correct run using `.item()` maintains constant memory.

#### Assignment 12.4 — Numerical Gradient Verification with `gradcheck`
- **Objective**: Validate a custom mathematical operation using PyTorch's official numerical gradient checker.
- **Given**: A custom smooth function with two inputs: $f(x, y) = x^2 y + \sin(x) \cos(y)$ in `torch.float64`.
- **Build / Do**:
  1. Test the function using `torch.autograd.gradcheck(f, (x, y), eps=1e-6, atol=1e-4)`.
  2. Introduce an intentional error in the analytical backward pass and re-run `gradcheck`.
  3. Observe and document the failure output.
- **Pass Criteria**:
  - The correct function passes `gradcheck`.
  - The intentionally flawed backward triggers a clear gradient discrepancy failure report.

#### Assignment 12.5 — Shuffled-Label Diagnosis
- **Objective**: Differentiate between architecture flaws and label noise using the permutation test.
- **Given**: A working classifier on a 2-class dataset.
- **Build / Do**:
  1. Run A: Train model on original dataset for 20 epochs.
  2. Run B: Train identical model on dataset with randomly permuted training labels: `shuffled_labels = labels[torch.randperm(len(labels))]`.
  3. In both runs, evaluate on uncorrupted validation data.
- **Pass Criteria**:
  - Show that Run B can still overfit training data (memorization) but validation accuracy remains at chance ($50\%$).
  - Explain why architecture tuning cannot fix a data-corruption issue.

#### Assignment 12.6 — Contextual Exception Handling in Data Loaders
- **Objective**: Enhance dataset debugging by enriching low-level exceptions with sample metadata.
- **Given**: A dataset loading files where one file is corrupted.
- **Build / Do**:
  1. Implement `__getitem__(idx)`:
     ```python
     try:
         return self.load_sample(idx)
     except Exception as err:
         raise RuntimeError(f"Error loading sample idx={idx}, path={self.paths[idx]}") from err
     ```
- **Pass Criteria**:
  - Traceback displays both the custom message naming the exact failing sample and the underlying low-level error.

**Exit criterion:** You follow an evidence-based debugging ladder from data to gradients instead of guessing.

---

## Chapter 13 — Accelerator execution and mixed precision

### What are we learning in this chapter?
- **Asynchronous GPU Execution**: How CUDA streams queue work and why naive timing with `time.time()` gives false benchmarks without `torch.cuda.synchronize()`.
- **The Host-Device Synchronization Trap**: Why calling `.item()` or `print()` inside the training loop forces the GPU to stall.
- **Automatic Mixed Precision (AMP)**: How using FP16/BF16 doubles throughput and cuts memory in half on modern accelerators.
- **The Float16 Underflow Problem & `GradScaler`**: Why tiny gradients vanish to zero in FP16, and how dynamic loss scaling prevents it.
- **The PyTorch GPU Memory Model**: Understanding Allocated memory vs. Caching Allocator Reserved memory, and what `empty_cache()` actually does.
- **Activation Checkpointing**: Trading a 20% increase in compute time to save 50% GPU memory on deep networks.

---

### 1. Asynchronous Execution: The Host and Device Queue

When you write:
```python
c = a + b  # a and b live on CUDA
```
The CPU does **not** wait for the GPU to finish adding the numbers!
Instead, the CPU sends a command to the GPU's command queue (the **CUDA stream**) and immediately moves on to execute the next line of Python code.

**Why Naive Timing Lies:**
```python
start = time.time()
output = model(x)
end = time.time()
print(f"Time: {end - start}s")  # WRONG! This only measures how fast the CPU queued the command!
```
To measure actual GPU execution time, you must wait for the GPU to finish:
```python
torch.cuda.synchronize()  # Wait for all queued GPU work to complete
start = time.time()
output = model(x)
torch.cuda.synchronize()  # Wait for forward pass to finish
end = time.time()
print(f"Real GPU Time: {end - start}s")
```

---

### 2. Automatic Mixed Precision (AMP) and `GradScaler`

Standard PyTorch computations use **Float32** (32 bits per number).
Modern GPUs (NVIDIA Tensor Cores) have specialized hardware that can run **Float16** and **BFloat16** up to 2–4x faster while consuming half the memory!

#### The Float16 Problem: Gradient Underflow
Float16 has only 5 exponent bits. Any number smaller than $5.96 \times 10^{-8}$ rounds to pure `0.0` (underflow).
In neural networks, gradient updates are often very small (e.g. $10^{-6}$). In Float16, these gradients round to zero and the model stops learning!

#### The Solution: `GradScaler`
1. Before backpropagation, multiply the loss by a large factor (e.g. $65,536$).
2. This scales all gradients up into the safe, representable Float16 range.
3. Compute backward pass.
4. Before updating weights, **unscale** the gradients back down by dividing by $65,536$.
5. If an overflow occurs (`Inf` or `NaN`), `GradScaler` discards the step, shrinks the scale factor, and tries again!

```python
from torch.amp import autocast, GradScaler
# Note: torch.autocast is the convenient top-level alias for torch.amp.autocast:
# with torch.autocast(device_type="cuda", dtype=torch.float16):

scaler = GradScaler("cuda")

for x, y in dataloader:
    x, y = x.to("cuda"), y.to("cuda")
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass runs in mixed precision (FP16/BF16)
    with autocast(device_type="cuda", dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)
        
    # Backward pass with scaled loss
    scaler.scale(loss).backward()
    
    # Optional: Unscale before gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step optimizer via scaler (skips update if Inf/NaN detected)
    scaler.step(optimizer)
    scaler.update()
```

---

### 3. GPU Memory: Allocated vs. Reserved

- **Allocated Memory (`torch.cuda.memory_allocated()`):** The exact bytes currently occupied by live PyTorch tensors.
- **Reserved Memory (`torch.cuda.memory_reserved()`):** The total memory PyTorch has requested from the OS.
PyTorch uses a **Caching Allocator**. When you delete a tensor (`del x`), PyTorch does not immediately return that memory to the operating system. It holds onto it so the next tensor allocation is instantaneous!
Calling `torch.cuda.empty_cache()` releases unused cached blocks back to the OS, but it **cannot** free memory used by active tensors.

---

### 4. Activation Checkpointing: Trading Compute for Memory

In deep networks, keeping intermediate activations in VRAM for the backward pass consumes up to 70% of GPU memory.
**Activation Checkpointing** (`torch.utils.checkpoint.checkpoint`) solves this:
- **Forward Pass:** Computes layers normally, but **discards** intermediate activations instead of saving them in memory.
- **Backward Pass:** As the backward pass reaches each block, it **recomputes** the forward activations on the fly!
- **Tradeoff:** Increases training time by ~20% (due to recomputation), but slashes activation memory by 50–70%, allowing you to fit 2–4x larger models or batch sizes in VRAM!

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([BigBlock() for _ in range(8)])
        
    def forward(self, x):
        for block in self.blocks:
            # use_reentrant=False is the recommended modern PyTorch 2 implementation:
            x = checkpoint(block, x, use_reentrant=False)
        return x
```

---

### Assignments 13

#### Assignment 13.1 — Fair Precision Benchmark (FP32 vs. AMP)
- **Objective**: Conduct a scientifically fair benchmark comparing FP32 and mixed-precision execution.
- **Given**: A deep CNN or small Transformer model, batch size 32.
- **Build / Do**:
  1. Run FP32 baseline:
     - Warmup: 10 steps.
     - Timed: 100 steps with `torch.cuda.synchronize()` before starting and stopping timers.
     - Measure: examples/sec, peak allocated memory, final loss.
  2. Run AMP (`torch.autocast` + `GradScaler`):
     - Warmup: 10 steps.
     - Timed: 100 steps with proper synchronization.
     - Measure: examples/sec, peak allocated memory, final loss.
- **Pass Criteria**:
  - Report: Throughput (samples/sec), peak allocated MB, and final loss for both.
  - Verify that AMP reduces peak memory and increases throughput while loss remains numerically comparable.

#### Assignment 13.2 — Float16 Underflow and `GradScaler` Mechanics
- **Objective**: Empirically demonstrate float16 gradient underflow and verify `GradScaler` compensation.
- **Given**: Small numbers representing gradients: $g = 10^{-5}$.
- **Build / Do**:
  1. Convert $g$ to `torch.float16`: show that successive multiplications by $0.1$ quickly underflow to `0.0`.
  2. Show that scaling by $2^{16}$ ($65,536$) keeps the value in the representable float16 range ($0.65536$).
  3. Simulate an overflow ($g = 10^5 \times 65,536 \implies \text{Inf}$):
     - Call `scaler.step(optimizer)` and verify that the optimizer step is skipped when gradients contain `Inf`.
- **Pass Criteria**:
  - Document the exact empirical underflow boundary for float16 ($< 5.96 \times 10^{-8}$).
  - Verify that `GradScaler` skips parameter updates when non-finite gradients are detected.

#### Assignment 13.3 — AMP with Scaled Gradient Norm Clipping
- **Objective**: Correctly integrate gradient norm clipping into an AMP training loop.
- **Given**: A model trained with `torch.autocast` and `GradScaler`.
- **Build / Do**:
  1. Inside the training step:
     ```python
     scaler.scale(loss).backward()
     scaler.unscale_(optimizer)  # Unscale before clipping
     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     scaler.step(optimizer)
     scaler.update()
     ```
  2. Test with normal gradients; assert clipping scales gradients when norm $> 1.0$.
  3. Test with an artificial `Inf` gradient; assert `scaler.step(optimizer)` detects overflow and skips update.
- **Pass Criteria**:
  - Assert that `grad_norm` reports the unscaled norm, not the scaled norm.
  - Verify optimizer update occurs when gradients are finite and is skipped when non-finite.

#### Assignment 13.4 — GPU Memory Allocation vs. Caching Allocator
- **Objective**: Inspect the distinction between allocated tensor memory and allocator-reserved memory.
- **Given**: CUDA-enabled environment.
- **Build / Do**:
  1. Record baseline: `memory_allocated()` and `memory_reserved()`.
  2. Allocate tensor: `x = torch.empty(1024, 1024, 256, device="cuda")` (~1 GB). Record stats.
  3. Delete tensor reference: `del x`. Record stats (allocated drops, reserved remains high).
  4. Call `torch.cuda.empty_cache()`. Record stats (reserved drops).
- **Pass Criteria**:
  - Tabulate the memory values at all 4 stages.
  - Explain why `empty_cache()` does not free memory occupied by active tensors.

#### Assignment 13.5 — Activation Checkpointing Tradeoff Study
- **Objective**: Measure the memory savings and runtime overhead of activation checkpointing.
- **Given**: A 4-layer Transformer model.
- **Build / Do**:
  1. Run 1 (Standard): Train for 50 steps without checkpointing. Record peak allocated memory and step time.
  2. Run 2 (Checkpointing): Wrap each transformer block in `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)`.
     - Train for 50 steps. Record peak allocated memory and step time.
- **Pass Criteria**:
  - Assert that outputs and gradients between Run 1 and Run 2 match within `1e-5`.
  - Report memory reduction (typically 30–60%) and time overhead (typically 15–30%).

#### Assignment 13.6 — Batch Size vs. Throughput Curve
- **Objective**: Find the optimal batch size for hardware utilization.
- **Given**: A fixed model and data pipeline.
- **Build / Do**:
  1. Test batch sizes $B \in \{8, 16, 32, 64, 128, 256\}$ up to GPU OOM.
  2. For each batch size, measure: throughput (samples/sec), step latency (ms), and peak memory (MB).
- **Pass Criteria**:
  - Plot or tabulate batch size vs. throughput.
  - Identify the saturation knee where throughput plateaus.

**Exit criterion:** You can conduct fair hardware benchmarks, implement AMP with GradScaler, and manage GPU memory.

---

## Chapter 14 — `torch.compile`: capture, guards, graphs, kernels

### What are we learning in this chapter?
- **Why `torch.compile` is Revolutionary**: How PyTorch 2 compilation eliminates Python interpreter overhead and fuses multiple operations into custom high-speed GPU kernels.
- **The Three Compilation Engines**: TorchDynamo (bytecode interceptor), AOTAutograd (ahead-of-time backward capture), and TorchInductor (C++/Triton code generator).
- **Guards & Graph Breaks**: Understanding why certain Python constructs (like `.item()` in an `if` condition) break compiled graphs, and how to debug them with `TORCH_LOGS="graph_breaks"`.
- **Dynamic Shapes**: Handling variable batch sizes and sequence lengths without triggering recompilation on every step (`dynamic=True`).
- **Numerical Parity**: Scientifically verifying that compiled models produce the exact same outputs and gradients as eager models.

---

### 1. What problem does `torch.compile` solve?

In standard (eager) PyTorch:
```python
y = torch.relu(x)
z = y + 2
```
1. Python executes line 1 $\implies$ launches a CUDA kernel to compute ReLU $\implies$ writes `y` back to GPU VRAM.
2. Python executes line 2 $\implies$ reads `y` back from GPU VRAM $\implies$ launches a second CUDA kernel to add 2 $\implies$ writes `z` to VRAM.

Notice the bottleneck: **Memory Bandwidth!**
The GPU spent more time reading and writing to memory than actually doing math!
With `torch.compile(model)`:
- PyTorch inspects your code ahead of time.
- It sees that ReLU and addition can be combined into a single operation.
- It generates a custom, fused GPU kernel in Triton that loads `x` once, applies ReLU and addition in fast GPU registers, and writes `z` once!
- Result: **1.3x to 2x speedup with zero code changes!**

```python
model = torch.compile(model)
```

---

### 2. The Three Engines Under the Hood

1. **TorchDynamo:** A Python frame evaluation hook that intercepts Python bytecode before execution. It separates pure PyTorch tensor operations from arbitrary Python code.
2. **AOTAutograd (Ahead-Of-Time Autograd):** Captures not just the forward pass, but also the backward pass graph *before* execution, allowing backward kernels to be optimized and fused.
3. **TorchInductor:** The backend compiler that translates the optimized graph into lightning-fast Triton code (for GPUs) or C++ code (for CPUs).

---

### 3. Graph Breaks: What They Are and How to Fix Them

A **Graph Break** occurs when Dynamo encounters Python code it cannot convert into a computation graph.
The most common culprit is a tensor-dependent Python branch:
```python
# GRAPH BREAK:
def forward(self, x):
    h = self.linear(x)
    if h.sum().item() > 0:  # .item() forces GPU to pause and send a Python float to CPU!
        return h * 2
    return h
```
When Dynamo sees `.item()`, it cannot know which branch will execute ahead of time. It has to stop compilation, fall back to slow Python, and start a second compiled graph afterwards.

**The Fix:** Use pure tensor operations:
```python
def forward(self, x):
    h = self.linear(x)
    return torch.where(h.sum() > 0, h * 2, h)  # No graph break! Fuses perfectly!
```

---

### 4. Dynamic Shapes (`dynamic=True`) and `TORCH_LOGS`

#### A. The Recompilation Problem & `dynamic=True`
By default, TorchInductor specializes GPU kernels for exact tensor dimensions (e.g. batch size 32).
If your batch size changes to 33, Inductor's **guards** fail. It must pause execution, re-invoke the compiler, and generate a brand-new kernel!
If your dataset has variable batch sizes or variable sequence lengths, this causes severe latency spikes.

**The Fix:** Tell PyTorch to treat dimensions as symbolic variables:
```python
# dynamic=True compiles a single kernel that accepts variable batch sizes!
compiled_model = torch.compile(model, dynamic=True)
```

#### B. Debugging with `TORCH_LOGS`
PyTorch 2 provides built-in logging flags to inspect what the compiler is doing:
- **`TORCH_LOGS="graph_breaks"`**: Prints the exact line of Python code that caused a graph break and why.
- **`TORCH_LOGS="recompiles"`**: Logs whenever a new kernel is compiled, along with the guard failure that triggered it.

You can set these in your shell before running:
```bash
TORCH_LOGS="graph_breaks,recompiles" python train.py
```
Or in Python:
```python
import torch._logging
torch._logging.set_logs(graph_breaks=True, recompiles=True)
```

---

### Assignments 14

#### Assignment 14.1 — `torch.compile` Benchmark: MLP and CNN
- **Objective**: Benchmark eager execution vs. compiled execution across warmup and steady-state phases.
- **Given**: An MLP and a CNN model, batch size 32.
- **Build / Do**:
  1. Compile both models with `compiled_model = torch.compile(model)`.
  2. Measure:
     - Eager step time (median over 100 steps).
     - First compiled call latency (capturing compilation overhead).
     - Compiled steady-state step time (median over 100 steps after warmup).
  3. Verify that outputs from eager and compiled models match before timing.
- **Pass Criteria**:
  - Assert `torch.testing.assert_close(eager_out, compiled_out)`.
  - Report eager time, first compiled call time, and compiled steady-state time.

#### Assignment 14.2 — Graph Break Diagnosis and Elimination
- **Objective**: Identify a graph break caused by Python control flow and eliminate it.
- **Given**: A module containing a tensor-dependent Python condition:
  ```python
  def forward(self, x):
      h = self.linear(x)
      if h.sum().item() > 0:  # Graph break: .item() forces Python scalar conversion
          return h * 2
      return h
  ```
- **Build / Do**:
  1. Run with `TORCH_LOGS="graph_breaks"` and observe Dynamo's log identifying the break.
  2. Rewrite the logic using pure tensor operations: `torch.where(h.sum() > 0, h * 2, h)`.
  3. Re-run and verify that the graph break is eliminated.
- **Pass Criteria**:
  - Log output confirms graph break on original code.
  - Log output confirms zero graph breaks on rewritten code.

#### Assignment 14.3 — Dynamic Shapes and Recompilation Audit
- **Objective**: Observe recompilation behavior across varying batch sizes and configure dynamic shape compilation.
- **Given**: Compiled model receiving inputs of varying batch sizes: $B \in \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$.
- **Build / Do**:
  1. Run 1: Compile with default settings `torch.compile(model)` with `TORCH_LOGS="recompiles"`. Feed inputs of varying batch sizes and count recompilation events.
  2. Run 2: Compile with `torch.compile(model, dynamic=True)`. Feed the same inputs and observe recompilations.
- **Pass Criteria**:
  - Document the number of compiled variants in Run 1 vs. Run 2.
  - Explain how dynamic shape guards prevent excessive kernel recompilations.

#### Assignment 14.4 — Eager vs. Compiled Numerical Parity Suite
- **Objective**: Rigorously verify numerical parity between eager and compiled models across forward and backward passes.
- **Given**: A 2-layer neural network with initial state saved to disk.
- **Build / Do**:
  1. Load identical weights into `model_eager` and `model_compiled = torch.compile(model)`.
  2. For 5 distinct random inputs:
     - Compare output logits: `torch.testing.assert_close(eager_out, comp_out, rtol=1e-4, atol=1e-4)`.
     - Compute loss and call `.backward()`.
     - Compare gradients for every parameter: `torch.testing.assert_close(param_eager.grad, param_comp.grad, rtol=1e-4, atol=1e-4)`.
- **Pass Criteria**:
  - Assertions pass for all 5 inputs across all parameter tensors.

#### Assignment 14.5 — Full Graph Compilation Test (`fullgraph=True`)
- **Objective**: Enforce complete graph capture with zero graph breaks using `fullgraph=True`.
- **Given**: The complete Transformer block from Chapter 10.
- **Build / Do**:
  1. Compile with `torch.compile(block, fullgraph=True)`.
  2. Execute forward and backward passes.
- **Pass Criteria**:
  - Execution completes successfully without raising `torch._dynamo.exc.Unsupported`.
  - If an error occurs, extract the minimal failing reproducer and document the unsupported feature.

**Exit criterion:** You understand how `torch.compile` captures and fuses kernels, how to diagnose and fix graph breaks, and when to use dynamic shapes.

---

## Chapter 15 — Profiling and performance engineering

### What are we learning in this chapter?
- **The Only Reliable Optimization Loop**: Measure $\to$ Identify Bottleneck $\to$ Form Hypothesis $\to$ Implement Isolated Change $\to$ Verify Correctness $\to$ Re-measure.
- **The PyTorch Profiler (`torch.profiler`)**: Setting up warmup/active schedules, recording shapes, and exporting Chrome trace files (`trace.json`) for timeline inspection.
- **The Power of Vectorization**: Quantifying the massive performance gap between Python loops and vectorized tensor operations ($100\times$ to $1,000\times$ speedup).
- **Data Pipeline Bottleneck Isolation**: Proving whether the GPU is starved by slow data loading using synthetic data baselines.
- **Host-Device Synchronization Stalls**: Eliminating frequent `.item()` and `print()` calls that cause the GPU to stall.

---

### 1. The Optimization Loop: Never Optimize Without Evidence

Never guess what makes your model slow. You will almost always guess wrong!
Always follow this scientific loop:
1. **Measure:** Profile the un-optimized code to establish a baseline (examples/second, GPU utilization, memory).
2. **Identify Bottleneck:** Find the single operation consuming the most time (e.g. data loading, memory bandwidth, host synchronization).
3. **Hypothesis:** "If I vectorize this distance loop using `torch.cdist`, step time will drop by 80%."
4. **Isolated Change:** Change *only* that one thing.
5. **Verify Correctness:** Assert that outputs and gradients are numerically identical before celebrating speedups!
6. **Re-measure:** Verify the speedup with synchronization and update your baseline.

---

### 2. Using `torch.profiler`

PyTorch has a world-class built-in profiler that traces both CPU and GPU activity:

```python
import torch

# Configure the schedule: wait 2 steps, warm up for 2 steps, record 3 steps
schedule = torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (x, y) in enumerate(dataloader):
        train_step(x, y)
        prof.step()  # Advances the profiler schedule

# Print top 10 most expensive GPU operations:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
You can load the resulting `trace.json` file into Chrome at `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) to see a visual timeline of every CUDA kernel executed!

---

### 3. Vectorization: The $100\times$ to $1000\times$ Speedup

Python loops in machine learning code are performance poison.
Consider computing all pairwise Euclidean distances between two sets of 1,000 vectors $X \in \mathbb{R}^{1000 \times 64}$ and $Y \in \mathbb{R}^{1000 \times 64}$:
- **Naive Python Loops:** Requires $1,000 \times 1,000 = 1,000,000$ iterations in the slow Python interpreter! (Takes seconds).
- **Vectorized PyTorch (`torch.cdist`):** Executes in a single C++/CUDA kernel in **under 1 millisecond**!
```python
# 1. Built-in vectorized distance function:
dist_vectorized = torch.cdist(X, Y)  # Shape [1000, 1000]

# 2. Or using broadcasted matrix algebra:
# ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x @ y.T
x_norm = (X**2).sum(dim=1, keepdim=True)  # [1000, 1]
y_norm = (Y**2).sum(dim=1, keepdim=True)  # [1000, 1]
dist_algebra = torch.sqrt((x_norm + y_norm.t() - 2 * (X @ Y.t())).clamp_min(0.0))
```

---

### 4. Eliminating Host-Device Synchronization Stalls

Remember: GPU execution is asynchronous!
If you call `.item()` on every single batch:
```python
# CATASTROPHIC PERFORMANCE PITFALL:
for x, y in dataloader:
    loss = train_step(x, y)
    losses.append(loss.item())  # Forces the CPU to halt and wait for the GPU on every step!
```
Every `.item()` call forces the CPU to stall until the GPU finishes calculating that exact loss value, destroying all concurrency between CPU and GPU.
**The Fix: Buffered / Periodic Logging:**
Keep losses on the GPU or only log every $N$ steps:
```python
total_loss = 0.0
for step, (x, y) in enumerate(dataloader):
    loss = train_step(x, y)
    total_loss += loss.detach()  # Keep on GPU!
    
    if (step + 1) % 20 == 0:
        # Synchronize only once every 20 steps:
        avg_loss = (total_loss / 20).item()
        print(f"Step {step+1} | Loss: {avg_loss:.4f}")
        total_loss = 0.0
```

---

### Assignments 15

#### Assignment 15.1 — Profiler Schedule & Timeline Trace Analysis
- **Objective**: Capture a multi-step profiler trace and identify the top compute, memory, and sync operations.
- **Given**: A training loop running a CNN or Transformer model.
- **Build / Do**:
  1. Configure `torch.profiler.profile`:
     ```python
     schedule = torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1)
     with torch.profiler.profile(
         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
         schedule=schedule,
         on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
         record_shapes=True,
         profile_memory=True,
         with_stack=True,
     ) as prof:
         for step, batch in enumerate(dataloader):
             train_step(batch)
             prof.step()
     ```
  2. Inspect trace and identify the 5 most expensive operations by self CUDA time and self CPU time.
- **Pass Criteria**:
  - Export `trace.json` file.
  - Produce a table listing the top 5 operations categorized by: Compute, Memory Movement, Host Overhead, or Synchronization.

#### Assignment 15.2 — Vectorization Speedup Quantification
- **Objective**: Quantify the performance difference between a Python loop and vectorized PyTorch tensor operations.
- **Given**: Pairwise distance computation between two sets of 1,000 vectors in $\mathbb{R}^{64}$.
- **Build / Do**:
  1. Implementation A: Nested Python loops calculating Euclidean distances.
  2. Implementation B: Fully vectorized implementation using `torch.cdist(x, y)` or broadcasted squared differences:
     $$\|x_i - y_j\|^2 = \|x_i\|^2 + \|y_j\|^2 - 2 x_i \cdot y_j$$
  3. Measure execution time of both implementations on CPU and GPU with proper synchronization.
- **Pass Criteria**:
  - Assert `torch.testing.assert_close(dist_loops, dist_vectorized, rtol=1e-4, atol=1e-4)`.
  - Report speedup factor (typically $> 100\times$ on CPU and $> 1,000\times$ on GPU).

#### Assignment 15.3 — Data Pipeline Tuning Isolation Experiment
- **Objective**: Systematically isolate and identify whether the dataloader is bottlenecking the GPU.
- **Given**: An end-to-end training loop.
- **Build / Do**:
  1. Synthetic data test: Replace dataloader with a dummy in-memory tensor generator (infinite loop yielding `torch.randn(...)`). Measure GPU throughput (baseline compute ceiling).
  2. Real data test: Measure throughput with real dataloader.
  3. If real throughput is significantly lower than synthetic throughput, systematically vary:
     - `num_workers`: 0 vs. 2 vs. 4 vs. 8
     - `pin_memory`: True vs. False
     - `persistent_workers`: True vs. False
- **Pass Criteria**:
  - Tabulate throughput across all configurations.
  - Determine whether the pipeline is compute-bound or data-loading-bound based on measured data.

#### Assignment 15.4 — Synchronization Overhead Quantification
- **Objective**: Measure the performance penalty of frequent `.item()` calls inside a GPU training loop.
- **Given**: A GPU training loop running 100 steps.
- **Build / Do**:
  1. Run 1 (Synchronous Logging): Call `loss.item()` at every single step:
     ```python
     for batch in dataloader:
         loss = train_step(batch)
         losses.append(loss.item())  # Forces GPU synchronization every step
     ```
  2. Run 2 (Buffered Logging): Keep losses on device or log only every 20 steps:
     ```python
     for step, batch in enumerate(dataloader):
         loss = train_step(batch)
         if step % 20 == 0:
             losses.append(loss.item())
     ```
  3. Time both runs using `torch.cuda.synchronize()` around the 100-step loop.
- **Pass Criteria**:
  - Report total elapsed time for Run 1 vs. Run 2.
  - Demonstrate a measurable throughput reduction in Run 1 and explain the GPU stall mechanism.

#### Assignment 15.5 — Structured Performance Engineering Report
- **Objective**: Synthesize profiling observations into a rigorous, professional engineering report.
- **Given**: An end-to-end training workload before and after optimization.
- **Build / Do**:
  - Write a 1-page performance report containing:
    1. Workload description & hardware/software specifications.
    2. Baseline performance metrics (throughput, latency, memory).
    3. Profiler evidence identifying the primary bottleneck.
    4. The isolated optimization applied (e.g. vectorization, AMP, compile, dataloader tuning).
    5. Correctness validation (numerical parity check).
    6. Post-optimization metrics and speedup factor.
- **Pass Criteria**:
  - All claims in the report are supported by empirical numbers and profiler evidence.

**Exit criterion:** Every optimization claim you make is backed by a representative profiler benchmark.

---

## Chapter 16 — Distributed training: DDP, FSDP2, TP

### What are we learning in this chapter?
- **The Scaling Continuum**: When and how to scale from a single GPU to Data Parallel (DDP), Fully Sharded Data Parallel (FSDP2), and Tensor Parallel (TP).
- **Communication Collectives from Scratch**: Understanding `all_reduce`, `all_gather`, and `reduce_scatter` in plain English.
- **DistributedDataParallel (DDP)**: How DDP replicates the model across GPUs and averages gradients asynchronously during backward.
- **Fully Sharded Data Parallel (FSDP2)**: How FSDP2 shards parameters, gradients, and optimizer states so you can train models that don't fit on one GPU.
- **Distributed Data Partitioning**: Using `DistributedSampler` to partition datasets without overlapping or dropping samples across ranks.
- **Exact Global Metric Reduction**: Correctly aggregating loss and accuracy across ranks processing unequal numbers of samples.

---

### 1. The Scaling Continuum: Which Strategy Do You Need?

```
+-------------------------------------------------------------------+
|  1. Does the model + batch fit on a SINGLE GPU?                   |
|     --> YES: Single-GPU training (keep it simple!)                |
+-------------------------------------------------------------------+
                                  | NO
                                  v
+-------------------------------------------------------------------+
|  2. Does the model fit on ONE GPU, but training is TOO SLOW?       |
|     --> Use DistributedDataParallel (DDP)                         |
|     (Replicates model on each GPU; each GPU trains on 1/N data)   |
+-------------------------------------------------------------------+
                                  | NO
                                  v
+-------------------------------------------------------------------+
|  3. Does the model NOT FIT on a single GPU (OOM with batch size 1)?|
|     --> Use FSDP2 (Fully Sharded Data Parallel)                   |
|     (Shards parameters, gradients, and optimizer across all GPUs) |
+-------------------------------------------------------------------+
                                  | STILL TOO BIG
                                  v
+-------------------------------------------------------------------+
|  4. Extreme scale (Models with 50B+ parameters):                  |
|     --> Use Tensor Parallel (TP) + FSDP2                          |
|     (Shards individual weight matrices across GPUs via DeviceMesh)|
+-------------------------------------------------------------------+
```

---

### 2. Communication Collectives: What Do They Do?

When training across $N$ GPUs (called **ranks**), they must communicate:

1. **`all_reduce(op=SUM)`:**
   - Every GPU has a number (e.g. Rank 0 has $1$, Rank 1 has $2$).
   - `all_reduce` calculates the sum ($1 + 2 = 3$) and gives the result **$3$ to every GPU**.
   - *Use case:* Averaging gradients across all GPUs after the backward pass.
2. **`all_gather`:**
   - Every GPU has a shard of a tensor (e.g. Rank 0 has shard A, Rank 1 has shard B).
   - `all_gather` collects all shards and gives the **full assembled tensor [A, B] to every GPU**.
   - *Use case:* In FSDP2, assembling a layer's full weights just before running its forward pass.
3. **`reduce_scatter`:**
   - Every GPU has a full gradient tensor.
   - It sums the gradients across all GPUs, then **splits the result into shards**, giving Rank 0 shard A and Rank 1 shard B.
   - *Use case:* In FSDP2, reducing gradients and sharding them across GPUs to save memory!
4. **`dist.barrier()`:**
   - A synchronization primitive that blocks execution until all ranks in the process group reach this point.
   - *Use case:* Ensuring Rank 0 finishes saving a checkpoint file to disk before other ranks attempt to load it, preventing race conditions and corrupted file reads.

---

### 3. DistributedDataParallel (DDP)

In DDP:
- Each GPU has its own independent Python process.
- Each GPU holds an identical copy of the entire model.
- Each GPU loads a different slice of data using `DistributedSampler`.
- During `loss.backward()`, DDP automatically triggers background `all_reduce` calls to average the gradients across all GPUs *while the rest of backward is still computing*!
- Every GPU steps its optimizer with the identical averaged gradients, so all models remain in perfect synchronization!

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Initialize process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2. Wrap model in DDP
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 3. Partition dataset
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Crucial for proper shuffling every epoch!
    for x, y in loader:
        ...
```

---

### 4. Testing Distributed Code on CPU with `torchrun` and Gloo

You don't need a cluster of GPUs to test distributed code!
PyTorch includes the **`gloo`** backend, which runs multi-process collective communication directly on CPU:

```python
import os
import torch
import torch.distributed as dist

def main():
    # 1. Initialize process group with Gloo for CPU testing:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 2. Each rank creates a tensor:
    t = torch.tensor([float(rank + 1)])  # Rank 0 has 1.0, Rank 1 has 2.0
    
    # 3. Sum across all ranks:
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert t.item() == 3.0
    
    # 4. Clean shutdown:
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```
Launch it from your terminal using `torchrun`:
```bash
torchrun --nproc_per_node=2 test_distributed.py
```

---

### 5. Exact Global Metric Reduction Across Unequal Ranks

Suppose you evaluate validation accuracy across 2 GPUs:
- **Rank 0:** Evaluates 10 samples, gets 8 correct (Accuracy = 80%).
- **Rank 1:** Evaluates 6 samples, gets 3 correct (Accuracy = 50%).

> [!WARNING]
> **The Rookie Mistake: Averaging Local Accuracies:**
> If you simply average local accuracies: $\frac{80\% + 50\%}{2} = 65\%$.
> This is **mathematically wrong**! Rank 0 processed more samples than Rank 1, so its score should have higher weight.
> The true global accuracy is:
> $$\frac{8 + 3}{10 + 6} = \frac{11}{16} = 68.75\%$$

**The Correct Way with `all_reduce`:**
Sum the raw counts of correct predictions and total samples across all ranks before dividing:
```python
local_correct = 8.0  # (or 3.0 on rank 1)
local_total = 10.0   # (or 6.0 on rank 1)

stats = torch.tensor([local_correct, local_total], dtype=torch.float64)
dist.all_reduce(stats, op=dist.ReduceOp.SUM)

global_correct = stats[0].item()
global_total = stats[1].item()
global_acc = global_correct / global_total
assert global_acc == 11.0 / 16.0
```

---

### Assignments 16

#### Assignment 16.1 — Multi-Process Collective Warm-Up with Gloo
- **Objective**: Launch a multi-rank distributed job and perform collective communication using the CPU Gloo backend.
- **Given**: A 2-process job launched via `torchrun --nproc_per_node=2`.
- **Build / Do**:
  1. Initialize process group: `dist.init_process_group(backend="gloo")`.
  2. Get rank and world size: `rank = dist.get_rank()`, `world_size = dist.get_world_size()`.
  3. Allocate tensor: `t = torch.tensor([float(rank + 1)])` (Rank 0 has 1.0, Rank 1 has 2.0).
  4. Perform all-reduce sum: `dist.all_reduce(t, op=dist.ReduceOp.SUM)`.
  5. Assert result is `3.0` on both ranks.
  6. Clean up: `dist.destroy_process_group()`.
- **Pass Criteria**:
  - Both ranks assert `t.item() == 3.0`.
  - Script completes cleanly without hanging or orphaned processes.

#### Assignment 16.2 — DDP Conversion and Sample Accounting
- **Objective**: Convert a single-process training script to DDP and verify exact dataset coverage.
- **Given**: A dataset of 100 samples with unique IDs `0..99`, 2 ranks, batch size 16.
- **Build / Do**:
  1. Wrap dataset in `DistributedSampler(dataset, shuffle=False)`.
  2. Wrap model in `torch.nn.parallel.DistributedDataParallel(model)`.
  3. In each rank, record the IDs of all processed samples over one epoch.
  4. Gather all processed IDs across ranks.
- **Pass Criteria**:
  - Show that every sample ID appears in the global multiset.
  - If `drop_last=False`, account for any padded samples added by `DistributedSampler`.

#### Assignment 16.3 — Global Metric Reduction Across Unequal Batches
- **Objective**: Compute exact global validation accuracy across ranks processing unequal numbers of samples.
- **Given**: Two ranks where Rank 0 evaluates 10 samples (8 correct) and Rank 1 evaluates 6 samples (3 correct).
- **Build / Do**:
  1. On each rank, maintain local tensors:
     - `local_correct = torch.tensor([num_correct], dtype=torch.float64)`
     - `local_total = torch.tensor([num_samples], dtype=torch.float64)`
  2. Perform `dist.all_reduce` on both tensors with `ReduceOp.SUM`.
  3. Compute global accuracy: `global_acc = global_correct / global_total`.
- **Pass Criteria**:
  - Assert that on both ranks, `global_acc` evaluates to exactly $\frac{8 + 3}{10 + 6} = \frac{11}{16} = 0.6875$.
  - Explain why averaging local accuracies ($\frac{0.8 + 0.5}{2} = 0.65$) is mathematically incorrect.

#### Assignment 16.4 — Distributed Checkpoint and Resume
- **Objective**: Save a checkpoint from rank 0 in a distributed run and resume training cleanly.
- **Given**: A 2-rank DDP training job.
- **Build / Do**:
  1. Train for 2 epochs.
  2. On rank 0, save checkpoint: `torch.save({"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)`.
  3. Synchronize ranks: `dist.barrier()`.
  4. In a new process, initialize DDP, load state dict into `model.module`, and verify both ranks have synchronized weights.
- **Pass Criteria**:
  - Assert that both ranks have identical parameter values after loading.
  - Resumed predictions on fixed inputs match across ranks.

#### Assignment 16.5 — Scaling Efficiency Benchmark (Multi-GPU)
- **Objective**: Measure throughput scaling efficiency from 1 GPU to multiple GPUs.
- **Given**: Multi-GPU environment.
- **Build / Do**:
  1. Benchmark throughput on 1 GPU: $T_1$ (samples/sec).
  2. Benchmark throughput on $N$ GPUs with identical per-device batch size: $T_N$.
  3. Calculate scaling efficiency:
     $$\text{Efficiency} = \frac{T_N}{N \times T_1} \times 100\%$$
- **Pass Criteria**:
  - Report $T_1, T_N$, and scaling efficiency percentage.
  - Identify communication overhead contributing to any efficiency drop below $100\%$.

#### Assignment 16.6 — FSDP2 Transformer Sharding
- **Objective**: Shard a multi-layer Transformer using PyTorch's FSDP2 API and compare memory footprint.
- **Given**: A multi-layer Transformer model on CUDA.
- **Build / Do**:
  1. Shard model bottom-up by wrapping individual `TransformerBlock` modules with `fully_shard`.
  2. Inspect local parameter shards: verify parameters are wrapped in `DTensor` representing shards.
  3. Measure peak allocated memory per rank compared to standard DDP.
- **Pass Criteria**:
  - Verify parameter sharding reduces per-GPU model memory proportional to rank count.
  - Assert that output and loss match standard un-sharded execution.

#### Assignment 16.7 — DDP vs. FSDP2 Execution Timeline Diagram
- **Objective**: Draw and explain the communication and memory lifecycle differences between DDP and FSDP2.
- **Given**: A 2-layer model trained across 2 ranks.
- **Build / Do**:
  - Construct a structured timeline diagram detailing parameter ownership, communication collective calls, and memory states:
    - Before forward
    - During forward
    - Before backward
    - During backward
    - Optimizer step
- **Pass Criteria**:
  - Diagram correctly places:
    - DDP: replicated parameters, `all_reduce` on gradients during backward.
    - FSDP2: sharded parameters, `all_gather` before forward, discard activations, `all_gather` before backward, `reduce_scatter` on gradients during backward, sharded optimizer step.

**Exit criterion:** You choose a parallelism strategy based on model and hardware constraints, and can reason about data ownership, state sharding, and communication collectives.

---

## Chapter 17 — Export, inference, and extension points

### What are we learning in this chapter?
- **Training State vs. Inference Artifacts**: Stripping away optimizers, gradient graphs, and dropout to produce lean deployment packages.
- **Secure Model Serialization**: Using `torch.save` and `torch.load` with `weights_only=True` to prevent arbitrary code execution vulnerabilities.
- **Ahead-of-Time Graph Export (`torch.export`)**: Capturing a self-contained, framework-independent `ExportedProgram` with dynamic shape specifications.
- **Cross-Platform ONNX Deployment**: Exporting models to ONNX and verifying bit-for-bit numerical parity in ONNX Runtime.
- **Custom Autograd Operators**: Writing custom `torch.autograd.Function` classes with analytical forward and backward passes, verified with `gradcheck`.
- **Vectorized Function Transforms (`torch.func`)**: Using `vmap` and `grad` to compute per-sample gradients without slow Python loops.

---

### 1. Training State vs. Inference Artifact

During training, a model is surrounded by massive scaffolding:
- Gradient graphs (`grad_fn`)
- Optimizer momentum buffers (often 2x the model's weight size!)
- Learning rate schedulers
- Stochastic layers like Dropout

For deployment (in a web server, mobile app, or robot), **none of that should exist**!
All you want is:
$$\text{Input Tensor} \implies \text{Model Weights} \implies \text{Output Prediction}$$

**The Deployment Checklist:**
1. Call `model.eval()`.
2. Wrap in `with torch.inference_mode():`.
3. Save only the `state_dict()` using `torch.save(model.state_dict(), "model.pt")`.
4. Load with `weights_only=True`:
   ```python
   # SECURE LOADING: Prevents malicious pickle execution!
   state_dict = torch.load("model.pt", map_location="cpu", weights_only=True)
   model.load_state_dict(state_dict)
   ```

---

### 2. `torch.export`: Ahead-of-Time Graph Capture

PyTorch 2 introduces `torch.export`—the official tool for exporting PyTorch models into standalone graph programs that can run outside of Python (e.g. in C++ runtimes or on edge devices):

```python
import torch

# 1. Define dynamic shape constraints (e.g. batch size can vary from 1 to 64)
batch_dim = torch.export.Dim("batch", min=1, max=64)
dynamic_shapes = {"x": {0: batch_dim}}

# 2. Export the model
exported_program = torch.export.export(
    model,
    args=(torch.randn(1, 16),),
    dynamic_shapes=dynamic_shapes,
)

# 3. Save the exported artifact to disk
torch.export.save(exported_program, "model.pt2")
```

---

### 3. Custom Autograd Functions: `torch.autograd.Function`

If you design a custom mathematical operation or write a custom CUDA kernel, you can hook it directly into PyTorch's autograd engine by subclassing `torch.autograd.Function`:

```python
class MySiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input tensor needed for backward pass
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        # Analytical derivative: d(x*sig)/dx = sig * (1 + x * (1 - sig))
        dx = sig * (1.0 + x * (1.0 - sig))
        return grad_output * dx
```

---

### 4. Cross-Platform Deployment: ONNX Export

If you need to deploy your model in a language other than Python (e.g. C++ for video games, C# for Windows apps, or Swift for iOS), export it to **ONNX (Open Neural Network Exchange)**:

```python
import torch

dummy_input = torch.randn(1, 16)

# Export model graph to ONNX format:
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    # Allow variable batch sizes:
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

To run inference in Python using the high-performance C++ `onnxruntime` engine:
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
ort_inputs = {"input": dummy_input.numpy()}
ort_outs = session.run(None, ort_inputs)
```

---

### 5. Vectorized Function Transforms (`torch.func`)

Standard PyTorch computes gradients for a whole batch summed together: $\nabla_\theta \sum_{i=1}^B \mathcal{L}(x_i, y_i)$.
What if you need **per-sample gradients** (e.g. for Differential Privacy, meta-learning, or sample importance analysis)?
- **Naive approach:** A Python `for` loop over individual samples $i = 0 \dots B-1$, calling `.backward()` on each one. This is agonizingly slow because it cannot take advantage of GPU tensor parallelism.
- **The Modern PyTorch 2 solution:** **`torch.func`** (formerly functorch), which brings composable functional transforms to PyTorch:
  1. **`torch.func.grad(fn)`:** Computes the gradient of a pure Python function with respect to its first argument.
  2. **`torch.func.vmap(fn, in_dims=...)`:** Vectorizes a function across a batch dimension automatically, executing in parallel without any Python loops!

```python
import torch
import torch.nn.functional as F

# 1. Define a stateless model forward + loss for a SINGLE sample:
def compute_loss(params, x, y):
    # params is a dictionary of parameter tensors: {"weight": W, "bias": b}
    pred = x @ params["weight"].t() + params["bias"]
    return F.cross_entropy(pred.unsqueeze(0), y.unsqueeze(0))

# 2. Derive gradient function for a single sample:
grad_fn = torch.func.grad(compute_loss)

# 3. Vectorize across the batch using vmap:
# in_dims=(None, 0, 0) means:
# - params is shared across all samples (None)
# - x is sliced along dimension 0 (batch dimension)
# - y is sliced along dimension 0 (batch dimension)
compute_per_sample_grads = torch.func.vmap(grad_fn, in_dims=(None, 0, 0))

# 4. Execute on a batch of 8 samples:
params = {
    "weight": torch.randn(4, 16, requires_grad=True),
    "bias": torch.zeros(4, requires_grad=True),
}
batch_x = torch.randn(8, 16)
batch_y = torch.randint(0, 4, (8,))

per_sample_grads = compute_per_sample_grads(params, batch_x, batch_y)
# per_sample_grads["weight"] has shape [8, 4, 16] -> exactly one gradient matrix per sample!
assert per_sample_grads["weight"].shape == (8, 4, 16)
assert per_sample_grads["bias"].shape == (8, 4)
```

---

### Assignments 17

#### Assignment 17.1 — Clean State Dict Round-Trip and Verification
- **Objective**: Serialize only model weights and restore them into a fresh model instance with verified prediction equality.
- **Given**: A trained model on GPU.
- **Build / Do**:
  1. Save weights: `torch.save(model.state_dict(), "model.pt")`.
  2. Instantiate fresh CPU model: `fresh_model = MyModel()`.
  3. Load weights safely: `fresh_model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))`.
  4. Set `fresh_model.eval()`.
  5. Pass identical input through original model (on CPU) and fresh model under `torch.inference_mode()`.
- **Pass Criteria**:
  - Assert exact prediction equality: `torch.testing.assert_close(model(x), fresh_model(x))`.
  - Verification that no optimizer or graph state was serialized.

#### Assignment 17.2 — `torch.export` with Dynamic Batch Dimension
- **Objective**: Export a model to an `ExportedProgram` with a dynamic batch dimension and validate shape constraints.
- **Given**: An MLP model taking input feature dimension $D = 16$.
- **Build / Do**:
  1. Specify dynamic batch dimension:
     ```python
     batch_dim = torch.export.Dim("batch", min=1, max=64)
     dynamic_shapes = {"x": {0: batch_dim}}
     exported = torch.export.export(model, (torch.randn(1, 16),), dynamic_shapes=dynamic_shapes)
     ```
  2. Test exported program with batch sizes $B \in \{1, 3, 16\}$.
  3. Test with invalid feature dimension $D = 15$ and show constraint error is raised.
- **Pass Criteria**:
  - Exported program executes successfully on batch sizes 1, 3, and 16.
  - Invalid feature dimension raises `torch.export.ConstraintViolationError`.

#### Assignment 17.3 — ONNX Export and Numerical Parity Suite
- **Objective**: Export a neural network to ONNX and verify output parity against eager PyTorch using ONNX Runtime.
- **Given**: A CNN or MLP classifier.
- **Build / Do**:
  1. Export model to ONNX:
     ```python
     torch.onnx.export(
         model,
         dummy_input,
         "model.onnx",
         input_names=["input"],
         output_names=["output"],
         dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
     )
     ```
  2. Load and run inference in `onnxruntime`: `session = ort.InferenceSession("model.onnx")`.
  3. Compare outputs across 5 distinct random inputs.
- **Pass Criteria**:
  - Assert maximum absolute error is within tolerance:
    `np.allclose(pytorch_out.numpy(), ort_out[0], rtol=1e-4, atol=1e-5)`.

#### Assignment 17.4 — Custom `autograd.Function` with `gradcheck`
- **Objective**: Implement a custom smooth mathematical activation function and verify its backward pass with `gradcheck`.
- **Given**: Smooth function $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ (Swish/SiLU).
- **Build / Do**:
  1. Analytical derivative: $f'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$.
  2. Define `class MySiLU(torch.autograd.Function)`:
     - `forward(ctx, x)`: save $x$ using `ctx.save_for_backward(x)`. Return $x \cdot \sigma(x)$.
     - `backward(ctx, grad_output)`: retrieve $x$, compute $f'(x)$, return `grad_output * f'(x)`.
  3. Run `torch.autograd.gradcheck(MySiLU.apply, (x,), eps=1e-6, atol=1e-4)` in `float64`.
- **Pass Criteria**:
  - `gradcheck` returns `True`.
  - Assert `torch.testing.assert_close(MySiLU.apply(x), F.silu(x))`.

#### Assignment 17.5 — Efficient Per-Sample Gradients via `torch.func`
- **Objective**: Compute individual sample gradients across a batch using `torch.func.vmap` and `torch.func.grad`.
- **Given**: A model, loss function, and batch of 8 samples `(X, y)`.
- **Build / Do**:
  1. Implementation A (Manual Loop): Loop over individual samples $i = 0 \dots 7$, compute loss, call `.backward()`, and collect per-sample parameter gradients.
  2. Implementation B (`torch.func`):
     - Define compute loss function for single sample: `loss_fn(params, x, y)`.
     - Compute per-sample gradient function: `grad_fn = torch.func.grad(loss_fn)`.
     - Vectorize across batch: `per_sample_grads = torch.func.vmap(grad_fn, in_dims=(None, 0, 0))(params, X, y)`.
- **Pass Criteria**:
  - Assert that `per_sample_grads` has leading dimension 8 (one gradient vector per sample).
  - Assert `torch.testing.assert_close(grads_loop, grads_vmap)`.

#### Assignment 17.6 — Comprehensive Inference Contract Specification
- **Objective**: Author an explicit, unambiguous inference contract document for production deployment.
- **Given**: A trained model ready for deployment.
- **Build / Do**:
  - Write an inference contract specification detailing:
    1. Framework & runtime version.
    2. Input tensor specifications: name, shape, dtype, valid numerical range, dynamic dimensions.
    3. Required preprocessing: normalization formula, color channel order, tokenization.
    4. Output tensor specifications: name, shape, dtype, interpretation (logits vs. probabilities).
    5. Numerical tolerance threshold for regression/classification.
    6. Minimal runnable Python verification script.
- **Pass Criteria**:
  - Specification is complete and unambiguous, allowing an independent engineer to run inference without inspecting training code.

**Exit criterion:** You can distinguish serialization, graph export, ONNX interchange, custom autograd operators, and functional transforms.

---

# Part V — Mastery Projects

## Chapter 18 — Projects that force retrieval

### What are we learning in this chapter?
- **End-to-End System Synthesis**: Combining data pipelines, modular architectures, training loops, profiling, and export into complete, robust systems.
- **The Six Verifiable Milestones**: A bulletproof project development roadmap from initial data exploration to closed-book reconstruction.
- **Four Production-Grade Capstones**:
  - Capstone A: Image Classifier as a Reliable System.
  - Capstone B: Character-Level Language Model (Decoder-Only Transformer).
  - Capstone C: Tabular Model and Interpretability-by-Ablation.
  - Capstone D: Distributed Scaling Lab (DDP & FSDP2).
- **The Capstone Acceptance Review**: The master checklist of questions every ML engineer must be able to answer about their system.

---

### Work in Verifiable Milestones

When building a complete machine learning project from scratch, do not write 1,000 lines of code and hit "run".
Follow these six verifiable milestones in strict order:

1. **Milestone 1: Data Contract & Exploration**
   - Inspect 3 individual raw samples and 1 assembled batch.
   - Assert shapes, dtypes, value ranges, and absence of `NaN`/`Inf`.
   - Explain the semantic meaning of every tensor dimension before touching model code.
2. **Milestone 2: Single Update Proof**
   - Feed 1 batch through the model.
   - Verify loss is finite, compute backward, verify `.grad` exists on all trainable parameters, and verify parameters move after `optimizer.step()`.
3. **Milestone 3: Tiny-Subset Overfit Proof**
   - Take 32 samples with augmentation disabled.
   - Train until the model achieves near-zero loss ($> 95\%$ accuracy).
   - If this fails, stop immediately! Your architecture or loss formulation has a bug.
4. **Milestone 4: End-to-End Baseline & Checkpoint Resumption**
   - Train on the full dataset, evaluate on validation data, save checkpoints, and prove that resumed training produces identical results.
5. **Milestone 5: One Measured Extension**
   - Identify your bottleneck using `torch.profiler`.
   - Implement one targeted improvement (e.g. AMP, `torch.compile`, or data pipeline tuning) and measure the speedup.
6. **Milestone 6: Closed-Book Rebuild**
   - One week later, reconstruct the core model and training loop from memory.

---

### Capstone Project Options

#### Capstone A — Image Classifier as a Reliable System
Build a complete image classification pipeline for a real or synthetic dataset from an empty directory:
- Deterministic train/validation/test split.
- Dataset validation and batch statistics visualization.
- Baseline CNN vs. pretrained model comparison.
- Tiny-subset overfit test.
- Checkpoint/resume and best-model selection.
- Profiler-backed performance tuning.
- Exported inference artifact (`torch.export` or ONNX) with a written inference contract.
- One-command training CLI and one-command inference CLI.

#### Capstone B — Character-Level Language Model
Build a decoder-only autoregressive transformer from scratch:
- Character tokenizer with reversible encode/decode round trip.
- Causal multi-head attention using `F.scaled_dot_product_attention`.
- Learnable positional embeddings and Pre-LN Transformer blocks.
- Weight tying between token embeddings and classification head.
- Temperature and top-k text generation sampler.
- Exact checkpoint resume including optimizer, scheduler, and RNG state.
- Optional KV cache for fast inference.

#### Capstone C — Tabular Model and Interpretability-by-Ablation
Build a classification or regression pipeline from raw CSV data:
- Schema validation, missing-value handling, and categorical encoding.
- Preprocessing fit strictly on the training split.
- Simple linear baseline compared against MLP.
- Imbalance-aware evaluation metrics (e.g. AUROC, F1).
- Feature ablation study measuring performance drop per feature across multiple random seeds.
- Compact CPU inference export.

#### Capstone D — Scaling Lab
Scale a model across multiple processes and GPUs:
- Single-device baseline measurement.
- Multi-rank DDP training with `DistributedSampler`.
- FSDP2 sharded transformer comparison.
- Exact global metric reduction across unequal batches.
- Distributed checkpointing.
- Profiler trace analyzing compute vs. communication balance.
- Scaling efficiency report.

---

### Capstone Acceptance Review Checklist

Before declaring your project finished, you should be able to answer every question without hand-waving:
- What does each axis of every boundary tensor mean?
- Why is this loss function mathematically correct for the target representation?
- Can the model overfit a tiny 32-sample subset?
- Which parameters and buffers change during training?
- What exactly is saved in the checkpoint, and can training resume identically?
- How was data leakage prevented between splits?
- What is the actual hardware bottleneck, according to profiler evidence?
- What breaks if the batch size or sequence length changes?
- Is evaluation using both `model.eval()` and `torch.inference_mode()`?
- Can a new process reproduce inference from the saved artifact?

---

### Assignments 18

#### Assignment 18.1 — Milestone 1: Data Contract & Exploration
- **Objective**: Establish and verify the data boundary contract before constructing model architectures.
- **Given**: Target dataset for your chosen capstone.
- **Build / Do**:
  1. Write a script that loads and displays 3 individual samples and 1 assembled batch.
  2. Implement programmatic assertions for every boundary field: tensor shape, dtype, value range, and absence of `NaN`/`Inf`.
  3. Document the semantic meaning of every tensor axis in the batch.
- **Pass Criteria**:
  - Script prints sample inspection and passes all assertions without error.

#### Assignment 18.2 — Milestone 2: Single-Batch Gradient Update Proof
- **Objective**: Verify that loss is finite, required gradients exist, and parameters update after one step.
- **Given**: Initialized model and one minibatch.
- **Build / Do**:
  1. Compute forward pass and loss; assert `torch.isfinite(loss)`.
  2. Call `loss.backward()`.
  3. Assert that every trainable parameter has a non-null, finite `.grad`.
  4. Record parameter values, call `optimizer.step()`, and assert that parameter values changed.
- **Pass Criteria**:
  - Script completes with all assertions passing.

#### Assignment 18.3 — Milestone 3: Tiny-Subset Overfitting Proof
- **Objective**: Prove architectural capacity and optimization viability by overfitting a 32-sample subset.
- **Given**: Exactly 32 samples from the dataset with augmentation disabled.
- **Build / Do**:
  1. Train for up to 200 iterations.
  2. Monitor training loss and accuracy.
- **Pass Criteria**:
  - Training accuracy reaches $> 95\%$ or MSE drops below $0.01$.

#### Assignment 18.4 — Milestone 4: End-to-End Baseline & Checkpoint Resumption
- **Objective**: Train a full baseline model and prove exact checkpoint resumption.
- **Given**: Full training and validation dataset.
- **Build / Do**:
  1. Train for $N$ epochs; save checkpoint at epoch $K$.
  2. In a fresh process, load checkpoint and resume training to epoch $N$.
  3. Compare predictions and metrics against uninterrupted run.
- **Pass Criteria**:
  - Validation metrics match uninterrupted reference within documented tolerance.

#### Assignment 18.5 — Milestone 5: One Measured Extension
- **Objective**: Implement one performance or architectural improvement justified by profiler evidence.
- **Given**: Baseline model and profiling trace from Milestone 4.
- **Build / Do**:
  1. Identify primary bottleneck from profiler.
  2. Implement one targeted change: AMP, `torch.compile`, architectural modification, or dataloader optimization.
  3. Benchmark before and after under identical conditions.
- **Pass Criteria**:
  - Document performance/correctness tradeoff with empirical numbers.

#### Assignment 18.6 — Milestone 6: Closed-Book Rebuild
- **Objective**: Solidify mastery by rebuilding the core system from memory.
- **Given**: An empty directory and your input/output data contract.
- **Build / Do**:
  - One week after completing the project, reconstruct the dataset, model, and training loop without consulting notes.
- **Pass Criteria**:
  - Code compiles, trains, and converges to baseline accuracy.

---

# Retrieval Ladder: 60 drills

Use these as short closed-book exercises. Do five per session.

## Level 1 — Tensor fluency

1. Create, reshape, transpose, and reduce a tensor; predict strides.
2. Normalize an image batch per channel.
3. Compute pairwise distances without loops.
4. Explain four broadcasting cases before running them.
5. Implement stable softmax by subtracting the maximum.
6. Implement one-hot encoding with indexing.
7. Select top-k values and recover original indices.
8. Build a causal boolean mask.
9. Pad variable-length sequences and build a padding mask.
10. Explain when `reshape` may copy.

## Level 2 — Gradients and modules

11. Differentiate a scalar polynomial with autograd.
12. Derive and verify linear-regression gradients.
13. Explain why `.grad` accumulates.
14. Implement `MyLinear`.
15. Count module parameters and buffers.
16. Freeze a backbone correctly.
17. Demonstrate `eval()` versus `inference_mode()`.
18. Find a detached gradient path.
19. Check a gradient with finite differences.
20. Make a residual block match dimensions.

## Level 3 — Complete training

21. Write a correct one-batch update.
22. Write weighted epoch-loss aggregation.
23. Write an evaluation function.
24. Overfit 32 examples.
25. Save and restore model plus optimizer.
26. Add gradient accumulation.
27. Add a scheduler with correct step timing.
28. Diagnose softmax-before-cross-entropy.
29. Prove one optimizer step changes parameters.
30. Reproduce an experiment under a fixed environment.

## Level 4 — Architectures

31. Calculate convolution output shape.
32. Implement one-channel convolution.
33. Explain receptive field growth.
34. Implement scaled dot-product attention.
35. Split and merge attention heads.
36. Distinguish causal, padding, and loss masks.
37. Build a transformer block.
38. Align next-token inputs and targets.
39. Implement top-k sampling.
40. Explain quadratic attention memory.

## Level 5 — Performance and scale

41. Benchmark asynchronous device code correctly.
42. Add autocast and gradient scaling.
43. Clip scaled gradients correctly.
44. Measure peak allocated memory.
45. Find one profiler bottleneck.
46. Explain a compile guard and graph break.
47. Compare first-call and steady-state compile time.
48. Tune DataLoader workers empirically.
49. Explain global batch size under DDP.
50. Aggregate distributed metrics correctly.

## Level 6 — Advanced delivery

51. Launch two-process DDP.
52. Explain all-reduce versus reduce-scatter/all-gather.
53. Wrap transformer blocks with FSDP2 bottom-up.
54. Save distributed state intentionally.
55. Export with a dynamic batch dimension.
56. Compare eager and exported outputs.
57. Implement and gradient-check a custom autograd function.
58. Compute per-example gradients with `torch.func`.
59. Write a complete inference contract.
60. Rebuild one capstone’s minimal training path from memory.

---

# Common decisions: a compact field guide

| Situation | Start here | Move on when |
|---|---|---|
| New model idea | eager PyTorch, float32, one device | correctness is demonstrated |
| Training fails | inspect data, overfit tiny subset | pipeline passes invariants |
| Need speed | profile representative workload | bottleneck is identified |
| Need lower precision | autocast; scaler for float16 training | accuracy is verified |
| Model fits, need throughput | DDP | communication dominates or scale target met |
| Model state does not fit | FSDP2 | memory/communication measured |
| Need Python-free artifact | `torch.export` | target runtime contract tested |
| Need interchange | Dynamo-based ONNX export | target-runtime parity tested |
| Need custom derivative | compose ops first, then `autograd.Function` | existing autograd cannot express it |
| Need custom kernel | compose + compile first | profiler proves the need |

---

# What not to learn first

Postpone these until a project creates the need:

- writing C++/CUDA extensions;
- multi-node fault tolerance;
- pipeline and tensor parallel composition;
- custom compiler backends;
- quantization internals;
- hand-built experiment-tracking infrastructure;
- every optimizer and scheduler;
- every model family.

Mastery comes from repeatedly building the core loop and then adding one constraint at a time.

---

# Coverage audit against the reference cheatsheet

This table maps every major section of the requested [reference cheatsheet](https://rohitbandaru.github.io/notes/pytorch-cheatsheet/) to the lesson that teaches it. “Covered” means the course explains the idea and includes an example or practice task; it does not mean the reference text was copied.

| Reference topic | Taught here |
|---|---|
| Tensor creation methods | 1.2 |
| Tensor data types | 1.3 and Chapter 13 |
| Random generation and seeding | 1.5 and Chapter 7 |
| Device management | 1.4, 2.16, Chapter 13 |
| Broadcasting semantics | 2.4 |
| `expand`, `repeat`, `broadcast_to`, tiling idea | 2.4 and Assignment 2.2 |
| Basic, ellipsis, integer, and boolean indexing | 2.5–2.6 |
| Gather, scatter, and scatter reductions | 2.7 and Assignment 2.4 |
| Indexed tensor assignment and broadcast assignment | 2.8 |
| Batch/coordinate/diagonal/top-k indexing strategies | 2.9 |
| Matrix multiplication (`matmul`, `mm`, `bmm`, element-wise) | 2.10 |
| Common shape, combine, split, select, sort, and mask operations | 2.1–2.3 and 2.9 |
| Folding, unfolding, and sliding windows | 2.11 and Assignment 2.5 |
| Operations over one or multiple dimensions | 2.3 |
| `einsum` | 2.12 and Assignment 2.6 |
| Einops `rearrange`, `reduce`, and `repeat` | 2.13 and Assignment 2.8 |
| Einops `einsum`, layer forms/EinMix, `pack`/`unpack` | 2.13 |
| Common neural-network layer families | 4.2, Chapters 8–10 |
| Attention, grouped-query heads, causal/padding masks | Chapter 9 and cache discussion in Chapter 10 |
| Custom modules/layers and containers | 4.1 and Assignment 4.1–4.2 |
| Weight initialization | Parameters and initialization in Chapter 4 |
| Module versus functional interface | 4.3 |
| Loss functions | Logits, probabilities, losses, and Assignment 4.5 |
| Optimizers | Chapter 6, Optimizers |
| Basic gradients | Chapter 3 |
| Training and evaluation loops | Chapter 6 |
| Autoregressive and recurrent generation concepts | Chapters 10 and 4.2 |
| Dataset and DataLoader | Chapter 5 |
| Custom datasets | 5.1–5.3 and Assignment 5.2 |
| Loader workers and tuning | Worker model in Chapter 5 and Assignment 5.6 |
| Transforms and augmentation | 5.2 and Assignment 5.4 |
| Advanced/variable-length/streaming dataset patterns | 5.3, Variable-length collation, Assignment 5.5 |
| Saving/loading weights and checkpoints | Chapter 6, Checkpointing |
| Model modes, evaluation, freezing | Chapter 3 grad modes, Chapters 6 and 11 |
| Learning-rate scheduling | Chapter 6, Schedulers and Assignment 6.4 |
| Custom autograd functions | 17, Custom autograd and Assignment 17.4 |
| Mixed-precision training | Chapter 13 |
| Clipping, accumulation, higher derivatives, hooks | Chapters 3 and 6 |
| In-place versus out-of-place operations | 2.15 |
| `view`, `reshape`, `permute`, transpose, contiguity | 2.14 |
| `clone`, `detach`, device/dtype transfers | 2.14–2.16 and Chapter 3 |
| In-place indexed operations | 2.7–2.9 |
| Memory management and activation checkpointing | Chapter 13 |
| Performance profiling and timing | Chapter 15 |
| NaN/gradient/activation debugging | Chapter 12 |
| Error handling and contextual failures | Chapter 12, Error handling |
| Further practice/resources | Retrieval Ladder, capstones, and official references below |

The course additionally covers reproducible tests, CNN/transformer internals, transfer learning and LoRA, `torch.compile`, DDP/FSDP2, `torch.export`, modern ONNX export, `torch.func`, and custom-operator decision making.

---

# Official reference map

These are references, not substitutes for practice.

- [Requested PyTorch cheatsheet (coverage reference)](https://rohitbandaru.github.io/notes/pytorch-cheatsheet/)
- [PyTorch 2.14 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)
- [PyTorch 2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)
- [Official installation selector](https://pytorch.org/get-started/locally/)
- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/)
- [Tensor reference](https://docs.pytorch.org/docs/stable/tensors.html)
- [Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [`nn` API](https://docs.pytorch.org/docs/stable/nn.html)
- [Data utilities](https://docs.pytorch.org/docs/stable/data.html)
- [Saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Automatic mixed precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Distributed tutorials](https://docs.pytorch.org/tutorials/distributed.html)
- [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [`torch.export` tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
- [Modern ONNX export tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [`torch.func`](https://docs.pytorch.org/docs/stable/func.html)
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)

---

# Setup checklist

Use an isolated environment and the command produced by the official installation selector because the correct wheel depends on OS and accelerator runtime.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Then use the command from https://pytorch.org/get-started/locally/
```

For this course, Python 3.12–3.14 is a practical choice. PyTorch 2.14 provides Python 3.15 wheels, but `torch.compile` is not yet supported on Python 3.15.

Verify:

```python
import torch

print("PyTorch:", torch.__version__)
print("Accelerator available:", torch.accelerator.is_available())
print("Accelerator:", torch.accelerator.current_accelerator(check_available=True))

x = torch.randn(128, 128)
device = torch.accelerator.current_accelerator(check_available=True)
if device is not None:
    x = x.to(device)
assert torch.isfinite(x @ x.T).all()
```

## Validation scope of this revision

The revision was checked against PyTorch 2.14 documentation. Its Python blocks were syntax-checked, and selected CPU worked examples were executed using the available PyTorch 2.13.0 environment, including training, accumulation edge cases, checkpoint restoration, attention, decoder gradients, dynamic export, and custom derivatives. This is not a claim that every exercise or every PyTorch 2.14 backend was executed. CUDA AMP, multi-GPU FSDP2, and optional ONNX-runtime paths require their stated environments and separate validation.

## Start now

Do Chapter 0 Assignment 1 on paper, then write the baseline from memory. Do not read Chapter 1 until you can explain all four lines of the learning loop.
