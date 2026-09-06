# lr — released split protocol

LogisticRegression(max_iter=3000), train-fit StandardScaler, ADASYN on training rows (released protocol; seed 42). ROC-AUC from probabilities.

| Split | N | WetFrac | Accuracy | ROC-AUC | WetF1 | DryF1 | DryRecall |
|---|---|---|---|---|---|---|---|
| random | 526 | 0.838 | 0.954 | 0.989 | 0.972 | 0.875 | 0.988 |
| temporal | 764 | 0.798 | 0.938 | 0.892 | 0.961 | 0.849 | 0.857 |
| site | 581 | 0.644 | 0.797 | 0.755 | 0.862 | 0.612 | 0.449 |

Per-class (dry / wet):

- **random**: dry P/R/F1 0.785/0.988/0.875 (n=85), wet P/R/F1 0.998/0.948/0.972 (n=441)
- **temporal**: dry P/R/F1 0.841/0.857/0.849 (n=154), wet P/R/F1 0.964/0.959/0.961 (n=610)
- **site**: dry P/R/F1 0.959/0.449/0.612 (n=207), wet P/R/F1 0.764/0.989/0.862 (n=374)
