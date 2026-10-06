# Probability, Statistics & Loss Functions

Machine learning is inherently probabilistic. We do not write deterministic IF/THEN rules; we build models that estimate the probability of an outcome given noisy inputs.

## Random Variables & Expectations

A random variable $X$ is a variable whose possible values are numerical outcomes of a random phenomenon.

* **Discrete:** Can take on distinct values (e.g., rolling a die, classifying an image as cat/dog/bird). Described by a Probability Mass Function (PMF).
* **Continuous:** Can take on any value in a range (e.g., predicting a house price, measuring temperature). Described by a Probability Density Function (PDF).

### Expected Value (Mean)

The long-run average value of a random variable.

$$\mathbb{E}[X] = \sum x_i P(x_i) \quad \text{(Discrete)}$$

## Bayes' Theorem

Bayes' Theorem provides a mathematical framework for updating our beliefs based on new evidence.

### The Bayes' Formula

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

* **Intuition:** What is the probability of $A$ happening, given that $B$ has occurred? You take your initial belief $P(A)$ (the Prior), multiply it by how likely the evidence $B$ is given $A$ (the Likelihood), and divide by the overall probability of $B$.
* **ML Application:** The foundation for Naive Bayes classifiers (used in spam filtering) and Bayesian Neural Networks, which output probability distributions instead of fixed point estimates.

## Maximum Likelihood Estimation (MLE)

MLE is the principle that bridges statistics and machine learning training. It asks: "Given the data we observed, what are the best model parameters ($\theta$) that make this data the most probable?"

### The Concept

We want to maximize the probability of our dataset given our weights:

$$\theta^* = \arg\max_\theta P(\text{Data} \mid \theta)$$

Because multiplying many small probabilities causes computational underflow, we take the natural logarithm. Maximizing the Log-Likelihood is identical to minimizing the **Negative Log-Likelihood (NLL)**.

* **ML Application:** Every standard loss function in machine learning is just a specific formulation of minimizing the Negative Log-Likelihood.

## Continuous Loss: Mean Squared Error (MSE)

If we assume the target variables $y$ contain continuous Gaussian (Normal) noise:

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Applying MLE to a Gaussian distribution mathematically simplifies directly to the Mean Squared Error.

### The MSE Formula

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

* **Intuition:** It measures the average squared difference between the true value ($y_i$) and the model's prediction ($\hat{y}_i$). Squaring the error heavily penalizes massive outliers.
* **ML Application:** Regression tasks—predicting house prices, forecasting stock values, or predicting bounding box coordinates in YOLO.

## Discrete Loss: Cross-Entropy (BCE and CCE)

If we assume our targets are discrete categories (Bernoulli distribution for 2 classes, Multinoulli for 3+), applying MLE gives us Cross-Entropy. It measures the difference between the true probability distribution and the model's predicted distribution.

### Binary Cross-Entropy (2 classes)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Categorical Cross-Entropy (3+ classes, paired with Softmax)

$$\hat{y}_k = \frac{\exp(z_k)}{\sum_{j=1}^{C} \exp(z_j)}$$

$$\mathcal{L}_{\text{CCE}} = -\sum_{k=1}^{C} y_k \log(\hat{y}_k)$$

* **Intuition:** If the true class is $1$, but the model predicts a probability of $0.01$, $\log(0.01)$ is a large negative number. The negative sign at the front turns this into a massive positive loss penalty.
* **ML Application:** Image classification, NLP sentiment analysis, and LLM next-token prediction.
