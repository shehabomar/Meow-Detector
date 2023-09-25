# Meow Meow Detector

### Description:
- Implementing the Logistic Regression Classifier with Neural Networks mindset to detect cats from images.
- Librabries used: `Matplotlib`, `h5py`, `numpy`, `PIL` and `scipy`.
- We are given a dataset of cats images as h5 file and the task is to tell if the image is a cat or not.

<hr>

### The Process of The Learning Algorithm:

![image](https://github.com/shehabomar/Smurf-Detector/assets/68251508/c8e922aa-56b5-4a0c-99f4-c945e487ddb9)

<hr>

### Mathematical expression of the algorithm:

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$

$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 

$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:

$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{4}$$

### Optimization

The goal is to learn $w$ and $b$ by minimizing the cost function $J$. 
For a parameter $\theta$, the update rule is $\theta$ = $\theta$ - $\alpha$ $\text{ } d\theta$, where $\alpha$ is the learning rate.

### Predict
 We are able now to use w and b to predict the labels for a dataset X. Implement the `predict()` function. There are two steps to computing predictions:

1. Calculate $\hat{Y} = A = \sigma(w^T X + b)$

2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 
