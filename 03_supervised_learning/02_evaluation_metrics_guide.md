# 📊 Comprehensive Guide to Machine Learning Evaluation Metrics

Selecting the right evaluation metric is one of the most critical decisions in Machine Learning Engineering. An improperly chosen metric can mask critical model failures, reward trivial majority-class classifiers, or misalign algorithmic objectives with business impact.

This guide provides an exhaustive theoretical and practical reference for all primary evaluation metrics: mathematical derivations, real-world failure modes, and concrete guidelines on **when to use each metric**.

---

## 🧭 Master Metric Selection Matrix

| Problem Type | Data Characteristics | Recommended Primary Metric | Secondary / Diagnostic Metric | Avoid Using |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Classification** | Balanced classes (~50/50) | **Accuracy**, **ROC-AUC** | F1-Score, Log Loss | — |
| **Binary Classification** | Highly Imbalanced (e.g., fraud, cancer) | **PR-AUC (Average Precision)**, **F1-Score / $F_\beta$** | Precision, Recall, MCC | **Accuracy** (Accuracy Paradox) |
| **Binary Classification** | False Positives are catastrophic | **Precision**, **$F_{0.5}$-Score** | Specificity, PR-AUC | High-recall tuned thresholds |
| **Binary Classification** | False Negatives are catastrophic | **Recall (Sensitivity)**, **$F_2$-Score** | PR-AUC, Balanced Accuracy | Precision-only tuning |
| **Binary Classification** | Probability calibration matters | **Log Loss (Cross-Entropy)**, **Brier Score** | Reliability Curve / ECE | Hard label accuracy |
| **Multi-Class Classification** | Balanced classes | **Multi-Class Accuracy**, **Macro F1** | Confusion Matrix | — |
| **Multi-Class Classification** | Imbalanced classes | **Macro-Averaged F1**, **Balanced Accuracy** | Per-Class Precision & Recall | Micro-Averaged F1 / Accuracy |
| **Regression** | Normal residuals, no extreme outliers | **RMSE (Root Mean Squared Error)** | $R^2$, MAE | MAPE if values $\approx 0$ |
| **Regression** | Heavy-tailed noise / extreme outliers | **MAE (Mean Absolute Error)** | Median Absolute Error, Huber Loss | MSE / RMSE |
| **Regression** | Multiplicative / percentage growth | **RMSLE**, **MAPE / sMAPE** | MAE | Raw RMSE across orders of magnitude |
| **Regression** | Explaining variance / goodness-of-fit | **$R^2$**, **Adjusted $R^2$** | RMSE | Unadjusted $R^2$ with many features |
| **Ranking / Retrieval** | Top-$K$ relevant items | **NDCG@K**, **MAP@K** | Precision@K, MRR | Raw classification accuracy |
| **Clustering** | No ground truth labels | **Silhouette Score** | Davies-Bouldin, Calinski-Harabasz | Supervised classification metrics |

---

## 1. Classification Metrics

### 1.1 The Confusion Matrix

The foundation of all discrete classification evaluation:

$$\begin{array}{c|cc}
& \text{Predicted Positive } (\hat{y}=1) & \text{Predicted Negative } (\hat{y}=0) \\
\hline
\text{Actual Positive } (y=1) & \mathbf{TP} \text{ (True Positive)} & \mathbf{FN} \text{ (False Negative / Type II Error)} \\
\text{Actual Negative } (y=0) & \mathbf{FP} \text{ (False Positive / Type I Error)} & \mathbf{TN} \text{ (True Negative)}
\end{array}$$

- **Total Positives ($P$)**: $TP + FN$
- **Total Negatives ($N$)**: $FP + TN$
- **Total Population**: $TP + TN + FP + FN$

---

### 1.2 Accuracy & The Accuracy Paradox

#### Mathematical Formulation
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Theoretical Meaning
Measures the overall fraction of correct predictions across all classes.

#### When to Use
- When classes are **roughly balanced** (e.g., 48% positive, 52% negative).
- When the business costs of False Positives and False Negatives are approximately equal.

#### When It Fails: The Accuracy Paradox
In imbalanced datasets (e.g., fraud detection where 99.9% of transactions are legitimate and 0.1% are fraudulent):
- A trivial dummy model that predicts `y = 0` (Legitimate) for every single transaction achieves **99.9% Accuracy**.
- However, this model identifies **0% of fraudulent cases** ($Recall = 0$) and is useless in production.
- **Rule of Thumb**: Never use raw Accuracy as the primary metric when class prevalence is skewed beyond 60/40.

---

### 1.3 Precision (Positive Predictive Value)

#### Mathematical Formulation
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### Theoretical Meaning
Answers: *"Of all instances the model predicted as Positive, what percentage was actually Positive?"*
Measures the purity/trustworthiness of positive predictions.

#### When to Use
Use Precision when **False Positives (Type I errors) carry a severe cost**:
- **Spam Email Filtering**: Marking a critical business email or flight confirmation as spam ($FP$) is far worse than letting one spam message into the inbox ($FN$).
- **Content Moderation / YouTube Copyright Strikes**: Banning or flagging legitimate creators erroneously ($FP$) damages user trust and creator livelihoods.
- **Legal Sentencing / Judicial Convictions**: The Blackstone principle ("Better that ten guilty persons escape than that one innocent suffer") prioritizes high precision.
- **Search Engine Recommendations**: Users only see the top 5 results; showing irrelevant items destroys engagement.

---

### 1.4 Recall (Sensitivity / True Positive Rate)

#### Mathematical Formulation
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{TP}{P}$$

#### Theoretical Meaning
Answers: *"Of all actual Positive instances that exist in reality, what percentage did the model successfully find?"*
Measures the completeness/coverage of the positive class.

#### When to Use
Use Recall when **False Negatives (Type II errors) carry catastrophic consequences**:
- **Cancer & Disease Diagnosis**: Missing a patient who actually has a malignant tumor ($FN$) leads to delayed treatment and death. A False Positive simply triggers a confirmatory biopsy.
- **Fraud & Financial Crime Detection**: Missing a multi-million-dollar money laundering transaction ($FN$) causes direct financial loss and regulatory fines.
- **Factory Defect Detection / Nuclear Safety Alarms**: Failing to catch a structural crack in an aircraft engine ($FN$) causes fatal disasters.
- **Cybersecurity Intrusion Detection**: An undetected hacker breach ($FN$) compromises the entire corporate network.

---

### 1.5 Specificity (True Negative Rate) & Fall-Out (False Positive Rate)

#### Formulations
$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{TN}{N}$$
$$\text{False Positive Rate (FPR)} = 1 - \text{Specificity} = \frac{FP}{TN + FP}$$

#### When to Use
- **Specificity** measures how well the model identifies genuine negative samples. Crucial in clinical screening to ensure healthy individuals are not overwhelmed with false alarms.
- **FPR** serves as the horizontal axis ($x$-axis) in the ROC curve.

---

### 1.6 The Precision-Recall Trade-off & $F_1$-Score

Precision and Recall are fundamentally in tension:
- Lowering the decision threshold ($\tau \to 0$) classifies more samples as positive $\to$ Recall increases, but False Positives surge $\to$ Precision drops.
- Raising the decision threshold ($\tau \to 1$) classifies only highly confident samples as positive $\to$ Precision increases, but False Negatives surge $\to$ Recall drops.

#### $F_1$-Score Formulation
The $F_1$-score is the **harmonic mean** of Precision and Recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

#### Why the Harmonic Mean (Instead of Arithmetic Mean)?
The arithmetic mean $\frac{P + R}{2}$ allows an extreme imbalance to yield a deceivingly high score:
- If a trivial model has $\text{Precision} = 1.0$ and $\text{Recall} = 0.02$, its arithmetic mean is $\frac{1.0 + 0.02}{2} = 0.51$.
- Its harmonic mean is:
  $$F_1 = 2 \cdot \frac{1.0 \cdot 0.02}{1.0 + 0.02} = \frac{0.04}{1.02} \approx \mathbf{0.039}$$
- The harmonic mean penalizes extreme disparities heavily: $F_1 \to 0$ if either Precision or Recall approaches 0.

#### When to Use $F_1$
- When you need a single scalar metric that balances Precision and Recall on an imbalanced dataset.

---

### 1.7 Generalized $F_\beta$-Score

When business priorities do not value Precision and Recall equally, the $F_\beta$-score introduces a weighting factor $\beta > 0$:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}} = \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + FP + \beta^2 \cdot FN}$$

- **$\beta = 1.0$ ($F_1$-Score)**: Equal weight to Precision and Recall.
- **$\beta = 2.0$ ($F_2$-Score)**: Weighs **Recall twice as heavily as Precision**. Ideal for medical diagnosis, search discovery, and fraud detection where False Negatives are 2x worse than False Positives.
- **$\beta = 0.5$ ($F_{0.5}$-Score)**: Weighs **Precision twice as heavily as Recall**. Ideal for spam filtering, automated loan approval, and customer notification where False Positives irritate users.

---

### 1.8 Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$

#### Theoretical Meaning
Computes the arithmetic mean of recall obtained on each class individually.
- On a balanced dataset, Balanced Accuracy equals standard Accuracy.
- On an imbalanced dataset, a model predicting only the majority class achieves $0.5 \cdot (0 + 1) = \mathbf{50\%}$ Balanced Accuracy, exposing its trivial nature.

---

### 1.9 Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

#### Properties
- Pearson product-moment correlation coefficient between actual and predicted binary classifications.
- Range: $[-1, +1]$:
  - $+1$: Perfect prediction.
  - $0$: No better than random coin toss.
  - $-1$: Total disagreement (inverse prediction).
- **Gold Standard for Extreme Imbalance**: Incorporates all four cells of the confusion matrix symmetrically. If a model predicts all samples as negative, MCC yields 0 or undefined, whereas accuracy would be 99.9%.

---

### 1.10 Multi-Class Averaging Strategies (Macro vs Micro vs Weighted)

In multi-class problems ($C > 2$), metrics like Precision, Recall, and F1 must be aggregated across classes $c \in \{1, \dots, C\}$:

#### 1. Macro Averaging
$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_1^{(c)}$$
- Treats **all classes equally**, regardless of sample size.
- **When to Use**: When performance on minority classes is just as important as on majority classes (e.g., rare diseases, edge cases).

#### 2. Micro Averaging
Aggregates total $TP, FP, FN$ globally across all classes before computing the metric:
$$\text{Micro } F_1 = \frac{2 \sum TP_c}{2 \sum TP_c + \sum FP_c + \sum FN_c}$$
- For multi-class single-label problems, $\text{Micro } F_1 = \text{Micro Precision} = \text{Micro Recall} = \text{Accuracy}$.
- **When to Use**: When overall sample throughput is prioritized over rare classes.

#### 3. Weighted Averaging
$$\text{Weighted } F_1 = \sum_{c=1}^C \frac{N_c}{N} F_1^{(c)}$$
- Weights each class by its support (number of true instances $N_c$).
- Accounts for class distribution, but can mask poor performance on small minority classes.

---

### 1.11 ROC Curve & ROC-AUC (Receiver Operating Characteristic)

The ROC curve plots **True Positive Rate (Recall)** against **False Positive Rate ($1 - \text{Specificity}$)** across all possible decision thresholds $\tau \in [0, 1]$.

#### Properties of ROC-AUC
- **Scale-Invariant**: Measures the ranking quality of predictions rather than absolute probability values.
- **Probabilistic Interpretation**: ROC-AUC equals the probability that the model will rank a randomly chosen positive sample higher than a randomly chosen negative sample:
  $$\text{ROC-AUC} = P(\hat{s}(x^+) > \hat{s}(x^-))$$
- **Values**:
  - $1.0$: Perfect separation.
  - $0.5$: Random guess (diagonal line).
  - $< 0.5$: Systematically inverted predictions.

#### Crucial Limitation: Why ROC-AUC Misleads on Imbalanced Data
The FPR denominator is $TN + FP = N$. When the negative class is massive ($N = 1,000,000$), generating $10,000$ False Positives yields:
$$\text{FPR} = \frac{10,000}{1,000,000} = \mathbf{1\%}$$
The ROC curve appears outstanding ($\text{FPR} \approx 0.01$, high TPR), but the model actually produces $10,000$ False Positives against only a few hundred True Positives, meaning actual Precision is abysmal ($< 5\%$).

---

### 1.12 Precision-Recall Curve & PR-AUC (Average Precision)

The PR curve plots **Precision** ($y$-axis) against **Recall** ($x$-axis) across all possible decision thresholds.

#### When to Use PR-AUC Over ROC-AUC
- **Highly Imbalanced Data**: PR-AUC ignores True Negatives ($TN$), focusing strictly on $TP, FP, FN$.
- A surge in False Positives immediately collapses Precision and tanks the PR-AUC, exposing model flaws that ROC-AUC hides.
- **Baseline**: A random classifier has $\text{ROC-AUC} = 0.5$, but its baseline $\text{PR-AUC} = \text{Prevalence} = \frac{P}{P + N}$. On a 1% fraud dataset, random guessing yields $\text{PR-AUC} = 0.01$.

---

### 1.13 Log Loss (Cross-Entropy) & Brier Score

When models output probabilities $\hat{p}_i \in [0, 1]$, threshold-based metrics (Accuracy, F1) discard confidence information.

#### Binary Log Loss
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]$$
- Heavily penalizes confident wrong predictions (e.g., predicting $\hat{p} = 0.999$ when true $y = 0$).

#### Brier Score (Mean Squared Error on Probabilities)
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{p}_i)^2$$
- Range: $[0, 1]$ (lower is better).
- Strictly proper scoring rule decomposing into **Reliability** (calibration), **Resolution**, and **Uncertainty**.

---

## 2. Regression Metrics

### 2.1 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

- **Theoretical Interpretation**: Median regression / $L_1$ loss.
- **Units**: Identical units to the target variable $y$.
- **When to Use**: When the dataset contains known outliers and each unit of error scales linearly in business cost.
- **Limitation**: Derivative is non-smooth at 0 ($\pm 1$); does not penalize occasional large catastrophic errors disproportionately.

---

### 2.2 Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

- **Theoretical Interpretation**: Mean regression / $L_2$ loss derived from Gaussian maximum likelihood.
- **Units**: RMSE has identical units to $y$; MSE has squared units ($y^2$).
- **When to Use**: When large errors are disproportionately worse than small errors (e.g., being off by 10 units is 4x worse than being off by 5 units).
- **Limitation**: Vulnerable to extreme outliers; a single anomalous outlier will dominate the total RMSE.

---

### 2.3 Mean Absolute Percentage Error (MAPE) & sMAPE

$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- **Theoretical Interpretation**: Relative percentage error.
- **When to Use**: When communicating business KPIs to executive leadership (e.g., "our forecast is accurate to within 4.2%").
- **Critical Failure Mode**: If any true value $y_i = 0$, division by zero causes infinite or undefined error.
- **Asymmetry**: Penalizes over-forecasts ($y_i = 10, \hat{y}_i = 100 \implies 900\%$ error) more than under-forecasts ($y_i = 100, \hat{y}_i = 10 \implies 90\%$ error).
- **Solution (sMAPE)**:
  $$\text{sMAPE} = \frac{100\%}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

---

### 2.4 Root Mean Squared Logarithmic Error (RMSLE)

$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \ln(y_i + 1) - \ln(\hat{y}_i + 1) \right)^2}$$

- **When to Use**:
  - Targets spanning multiple orders of magnitude (e.g., house prices, sales counts, web traffic).
  - When relative error matters more than absolute dollar difference (under-predicting a \$10 product by \$5 is 50% error, but under-predicting a \$10,000 product by \$5 is negligible).
  - Penalizes under-estimates more severely than over-estimates.

---

### 2.5 Coefficient of Determination ($R^2$) & Adjusted $R^2$

$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

- **Interpretation**: Proportion of target variance explained by the model compared to a baseline model that always predicts the sample mean $\bar{y}$.
  - $R^2 = 1.0$: Perfect predictions.
  - $R^2 = 0.0$: Equivalent to predicting the mean $\bar{y}$.
  - $R^2 < 0.0$: Model is worse than predicting the mean!
- **Flaw of $R^2$**: Adding any new feature $x_{p+1}$ to a linear regression model *always* increases or maintains $R^2$, even if the feature is pure random noise.
- **Adjusted $R^2$ Fix**:
  $$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(N - 1)}{N - p - 1} \right]$$
  Penalizes the addition of uninformative predictors $p$.

---

## 3. Ranking & Information Retrieval Metrics

Used in Search Engines, Recommendation Systems, and RAG (Retrieval-Augmented Generation):

### 3.1 Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- Evaluates the position of the **first relevant item** in search results. If the first relevant result appears at rank 1, score = 1.0; at rank 2, score = 0.5.

### 3.2 Mean Average Precision (MAP@K)
$$\text{AP}@K = \frac{1}{\min(K, R)} \sum_{k=1}^K P@k \cdot \text{rel}(k)$$
$$\text{MAP}@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}_q@K$$
- Evaluates top-$K$ recommendations when items have binary relevance (relevant/not relevant).

### 3.3 Normalized Discounted Cumulative Gain (NDCG@K)
Evaluates graded relevance (e.g., ratings from 0 to 5) with logarithmic positional discounting:

$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
where $\text{IDCG}@K$ is the Ideal DCG achieved by an optimal sort of relevant items.

---

## 4. Clustering & Unsupervised Metrics

### 4.1 Silhouette Score
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: Mean intra-cluster distance from sample $i$ to all other points in the same cluster.
- $b(i)$: Mean nearest-cluster distance from sample $i$ to points in the closest neighboring cluster.
- Score $\in [-1, +1]$:
  - $\approx +1$: Sample is well-clustered and far from neighboring clusters.
  - $\approx 0$: Sample is on or near the decision boundary between two clusters.
  - $\approx -1$: Sample is misassigned to the wrong cluster.

### 4.2 Davies-Bouldin Index
$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \quad \text{DB} = \frac{1}{K} \sum_{i=1}^K \max_{j \ne i} R_{ij}$$
- Ratio of within-cluster scatter to between-cluster separation. **Lower values indicate better clustering**.

### 4.3 Calinski-Harabasz Index (Variance Ratio Criterion)
$$\text{CH} = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \cdot \frac{N - K}{K - 1}$$
- Ratio of between-cluster dispersion to within-cluster dispersion. **Higher values indicate tighter, well-separated clusters**.
