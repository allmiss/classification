| Метод | Macro F1 | Δ к baseline | Δ, % |
|---|---|---|---|
| Baseline: TF-IDF (1,2) + LogReg | 0.5415 | +0.0000 | +0.0% |
| N-grams (1,1) + LogReg | 0.5564 | +0.0149 | +2.7% |
| N-grams (1,3) + LogReg | 0.5436 | +0.0021 | +0.4% |
| CountVectorizer (1,2) + LogReg | 0.5659 | +0.0244 | +4.5% |
| TF-IDF char_wb (2,5) + LogReg | 0.5900 | +0.0485 | +9.0% |
| TF-IDF (1,2) + LinearSVC | 0.5539 | +0.0124 | +2.3% |
| TF-IDF (1,2) + ComplementNB | 0.6585 | +0.1169 | +21.6% |
| TF-IDF (1,2) + RandomForest | 0.4967 | -0.0449 | -8.3% |
| TF-IDF (1,2) + LogReg + GridSearch | 0.6509 | +0.1093 | +20.2% |
| TF-IDF char_wb (2,5) + LogReg + GridSearch | 0.6913 | +0.1497 | +27.6% |
