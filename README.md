# CI2024 – Project Work: Symbolic Regression

## Overview
This project implements an **evolutionary algorithm for symbolic regression**, aiming to discover mathematical expressions that best fit given datasets.  

Formulas are represented as **trees composed of**:
- Unary and binary NumPy operators
- Variables
- Constants in the range `[-5, 5]`

The project is tested on **8 datasets**, each with inputs of dimensionality 1–6 and one output.

---

## Methodology

### Evolutionary Loop
The algorithm evolves a population of candidate formulas using genetic operators:

- **Fitness function**: based on **mean squared error (MSE)** and **formula length** (penalizing overly complex expressions).
- **Selection**: tournament selection, balancing exploration and exploitation.
- **Crossover**: subtree swap between two parents, combining structural parts of both.
- **Mutation**:
  - **Random mutation**: replaces nodes with alternative operators, constants, or variables.
  - **Subtree mutation**: replaces an entire subtree with a new randomly generated one.
  - **Shrink mutation**: reduces tree size when complexity exceeds a threshold.

### Optimization Features
- **Adaptive mutation rate**: increases over time to promote exploration.
- **Population update**: only the best individuals are kept each generation.
- **Early stopping**: halts evolution if no significant improvement occurs.

---

### Problem 1
#### MSE: 7.125940794232773e-34
```python
return np.sin(x[0])
```

### Problem 2
#### MSE: 5957085970506.793
```python
return np.multiply(np.add(x[1], np.add(np.add(x[2], np.sin(x[0])), np.exp(np.multiply(1.40434, np.add(np.exp(1.59888), x[0]))))), np.add(np.subtract(np.cos(x[1]), np.add(x[1], np.add(x[2], 2.69074))), np.multiply(np.multiply(np.subtract(np.multiply(np.multiply(2.04716, np.multiply(1.68386, np.add(np.exp(1.68386), x[0]))), np.add(np.add(np.multiply(np.sin(x[2]), x[2]), np.add(np.cos(x[1]), np.exp(2.41153))), np.multiply(np.subtract(np.cos(np.subtract(1.1837, np.multiply(1.33387, x[0]))), np.add(x[1], np.add(x[1], np.add(x[2], np.add(x[2], np.add(1.74239, np.multiply(np.multiply(0.78674, np.add(np.exp(1.91645), x[0])), x[0]))))))), np.add(np.exp(1.63605), x[0])))), np.multiply(np.multiply(np.subtract(x[1], -2.74036), np.cos(x[0])), np.exp(np.add(x[2], x[0])))), -2.06294), np.exp(np.subtract(np.multiply(np.cos(-0.537669), np.subtract(0.958793, np.multiply(1.10898, x[0]))), x[0])))))
```

### Problem 3
#### MSE: 5.689164358710445e-08
```python
return np.add(np.mod(np.sin(np.sin(np.sin(np.square(-1.78261)))), np.mod(-1.78261, np.cosh(np.mod(-1.78261, np.cosh(np.mod(-1.78261, np.cosh(np.mod(-1.78261, np.cosh(np.mod(-1.78261, np.cosh(0.324447))))))))))), np.add(np.add(np.trunc(0.324447), np.multiply(0.989134, np.divide(np.add(x[2], -1.78261), -0.853707))), np.add(0.324447, np.add(np.divide(np.add(x[2], -1.02091), -0.853707), np.add(-1.95991, np.subtract(np.add(np.square(x[0]), np.add(np.square(x[0]), np.divide(np.add(x[2], -1.78261), -0.853707))), np.multiply(np.add(np.square(x[1]), np.multiply(np.divide(-0.531521, np.divide(x[1], x[2])), np.mod(-1.02091, np.mod(np.arccos(-0.311599), 0.0896656)))), x[1])))))))
```

### Problem 4
#### MSE: 2.1066255989255914e-08
```python
return np.add(np.mod(-0.522209, np.subtract(1.25057, np.sin(np.reciprocal(1.91346)))), np.add(np.add(np.add(np.add(np.subtract(1.25057, np.sin(np.divide(x[1], x[1]))), np.add(np.add(0.0155197, np.add(np.sin(1.46992), np.add(np.cos(x[1]), np.cos(x[1])))), np.cos(x[1]))), np.add(np.subtract(1.21274, np.add(-0.418273, np.multiply(np.cos(np.multiply(np.cos(np.reciprocal(np.reciprocal(0.53107))), np.multiply(np.cos(np.multiply(np.cos(0.631819), np.multiply(np.cos(np.multiply(np.cos(-1.46854), np.subtract(np.subtract(np.reciprocal(x[0]), np.divide(x[1], x[1])), np.subtract(np.divide(-1.46854, -0.381457), np.add(np.multiply(1.531, np.add(np.sin(-0.522209), np.sin(x[0]))), np.add(np.sin(x[0]), x[0])))))), np.multiply(0.0953769, np.divide(-1.46854, -0.381457))))), np.multiply(0.0953769, np.divide(-1.46854, -0.381457))))), np.multiply(0.0953769, x[0])))), np.cos(x[1]))), np.add(np.cos(x[1]), np.cos(x[1]))), np.cos(x[1])))
```

### Problem 5
#### MSE: 5.572810232617333e-18
```python
return np.multiply(np.sin(-0.40481), np.subtract(x[1], x[1]))
```

### Problem 6
#### MSE: 1.3738178509961727e-10 
```python
return np.subtract(np.multiply(1.00525, np.multiply(np.sin(np.subtract(1.14646, np.add(1.5973, -1.20983))), np.subtract(x[1], x[0]))), np.subtract(np.multiply(np.sin(1.14646), np.multiply(np.multiply(np.multiply(0.887104, np.cos(1.32699)), np.multiply(np.cos(0.0433145), np.subtract(x[0], np.multiply(np.cos(np.multiply(np.sin(np.sin(np.multiply(np.multiply(0.916361, np.cos(1.32699)), np.multiply(0.887104, np.subtract(np.multiply(x[0], 0.873967), np.add(np.add(np.sin(x[1]), np.sin(x[1])), x[1])))))), np.multiply(np.subtract(1.05642, np.cos(0.916361)), np.subtract(-0.148426, np.multiply(np.cos(1.00525), -0.318614))))), x[1])))), np.subtract(-0.148426, np.multiply(np.cos(1.00525), np.sin(np.multiply(np.cos(0.0433145), np.sin(np.sin(-0.318614)))))))), x[1]))
```

### Problem 7
#### MSE: 89.17723738108788
```python
return np.add(np.add(x[1], np.add(-0.286507, np.multiply(np.mod(x[1], np.multiply(np.negative(np.sqrt(1.94221)), np.mod(x[0], np.negative(np.multiply(-0.906251, np.negative(np.mod(x[1], np.multiply(np.mod(x[0], np.multiply(x[1], 1.20793)), 1.20793)))))))), np.subtract(x[0], np.multiply(np.add(1.20793, np.ceil(np.cosh(1.94221))), np.negative(np.add(np.floor(np.mod(x[1], np.positive(np.multiply(x[0], 1.20793)))), np.add(x[0], np.positive(np.mod(x[0], np.multiply(x[1], 1.20793))))))))))), np.add(np.cosh(np.subtract(-0.286507, np.add(np.floor(np.mod(x[1], np.positive(np.multiply(x[0], 1.20793)))), np.multiply(x[1], 1.25449)))), np.cosh(np.add(np.multiply(1.20793, np.rint(np.positive(np.multiply(x[0], 1.20793)))), np.add(np.mod(x[1], np.multiply(np.negative(np.sqrt(np.cosh(1.20793))), np.mod(x[1], np.negative(np.multiply(-0.906251, np.negative(np.mod(x[0], np.multiply(np.negative(np.sqrt(np.cosh(1.20793))), np.mod(x[0], np.negative(np.multiply(-0.906251, np.negative(x[1])))))))))))), np.mod(x[0], np.multiply(x[1], 1.20793)))))))
```

### Problem 8
#### MSE: 323132.3660634279
```python
return np.subtract(np.multiply(np.add(x[5], np.add(np.arctanh(-0.00788088), np.add(np.cbrt(np.sign(0.181233)), np.multiply(np.arctan(x[5]), np.mod(-1.93523, np.exp(x[5])))))), np.exp(np.tan(-1.79232))), np.mod(-1.31863, np.add(np.multiply(np.abs(x[5]), np.exp(np.tan(np.multiply(1.61589, 0.867438)))), np.add(np.cosh(np.subtract(np.multiply(np.sin(x[4]), np.sinh(1.04268)), np.multiply(-1.99056, x[4]))), np.power(np.tan(np.tan(np.tan(np.tan(np.tan(np.tan(np.tan(np.tan(0.136319)))))))), x[5])))))
```
