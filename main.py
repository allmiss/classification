import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from preprocess import LABEL2ID, preprocess_text

df = pd.read_csv("data.csv")
df = df.rename(columns={"Sentence": "text", "Sentiment": "sentiment"})

print(f"Размерность - {df.shape}")
print(f"Первые 5 строк:\n{df.head()}")
df.info()
print(f"Количество пропущенных значений:\n{df.isnull().sum()}")

print(df["sentiment"].value_counts())
print(df["sentiment"].value_counts(normalize=True))

plt.figure()
sns.countplot(data=df, x="sentiment")
plt.title("Classes spread")
plt.savefig("reports/sentiment_distribution.png")
plt.close()

df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
print(df[["text_length", "word_count"]].describe())
print(df.groupby("sentiment")["word_count"].mean())

plt.figure()
sns.histplot(data=df, x="text_length", bins=50)
plt.title("Len histogram")
plt.xlabel("Text length")
plt.ylabel("Text count")
plt.savefig("reports/text_length_distribution.png")
plt.close()

for sentiment in df["sentiment"].unique():
    print(f"\n=== Класс {sentiment} ===")
    for example in df[df["sentiment"] == sentiment]["text"].sample(3, random_state=42):
        print(f"Example - {example}")


df["text_clean"] = df["text"].apply(preprocess_text)

rows_before = len(df)
df = df[df["text_clean"].str.len() > 0]
print(f"Удалено пустых после очистки: {rows_before - len(df)}")

rows_before = len(df)
df = df.drop_duplicates(subset=["text_clean", "sentiment"])
print(f"Удалено полных дубликатов: {rows_before - len(df)}")

print(f"Осталось строк: {len(df)}")


df["label"] = df["sentiment"].map(LABEL2ID)
print(f"Неизвестных меток: {df['label'].isnull().sum()}")
print(df[["sentiment", "label"]].value_counts())


checks = {
    "ссылка": df["text"].str.contains("http"),
    "кавычки": df["text"].str.contains("``", regex=False),
    "пробел перед знаком": df["text"].str.contains(" , ", regex=False),
    "отдельное 's": df["text"].str.contains(" 's", regex=False),
}
for name, mask in checks.items():
    print(f"\n=== {name} ===")
    for _, row in df[mask].head(2).iterrows():
        print(f"ДО:     {row['text']}")
        print(f"ПОСЛЕ:  {row['text_clean']}")
        print()


train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\nTrain: {len(train_df)} строк, Test: {len(test_df)} строк")
print("Доли классов в train:")
print(train_df["label"].value_counts(normalize=True).sort_index())
print("Доли классов в test:")
print(test_df["label"].value_counts(normalize=True).sort_index())


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["text_clean"])
X_test = vectorizer.transform(test_df["text_clean"])
y_train = train_df["label"].values
y_test = test_df["label"].values

print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
print(f"Пример признако: {vectorizer.get_feature_names_out()[:15]}")


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nПервые 10 предсказаний: {y_pred[:10]}")
print(f"\nПервые 10 правильных: {y_test[:10]}")

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=list(LABEL2ID.keys())))


baseline_results = {
    "model": "TF-IDF (1,2), max_features=5000 + LogisticRegression",
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
    "train_size": len(train_df),
    "test_size": len(test_df),
}

with open("reports/baseline_metrics.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)

with open("reports/baseline_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, y_pred, target_names=list(LABEL2ID.keys())))

print("Baseline результаты сохранены в reports/")


def evaluate(name, vectorizer, model):
    X_train = vectorizer.fit_transform(train_df["text_clean"])
    X_test = vectorizer.transform(test_df["text_clean"])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"{name:45s} Macro F1: {f1:.4f}")
    return {
        "name": name,
        "f1": f1,
        "model": model,
        "vectorizer": vectorizer,
        "pred": pred,
    }


print("\n=== День 4: улучшения ===")
results = []
results.append(
    evaluate(
        "Baseline: TF-IDF (1,2) + LogReg",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, random_state=42),
    )
)

# Способ 1: другие n-граммы
results.append(
    evaluate(
        "N-grams (1,1) + LogReg",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 1)),
        LogisticRegression(max_iter=1000, random_state=42),
    )
)
results.append(
    evaluate(
        "N-grams (1,3) + LogReg",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 3)),
        LogisticRegression(max_iter=1000, random_state=42),
    )
)
# Способ 2: другой екторизатор
results.append(
    evaluate(
        "CountVectorizer (1,2) + LogReg",
        CountVectorizer(max_features=5000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, random_state=42),
    )
)
results.append(
    evaluate(
        "TF-IDF char_wb (2,5) + LogReg",
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=20000),
        LogisticRegression(max_iter=1000, random_state=42),
    )
)
# Способ 3: другая модель
results.append(
    evaluate(
        "TF-IDF (1,2) + LinearSVC",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        LinearSVC(random_state=42),
    )
)
results.append(
    evaluate(
        "TF-IDF (1,2) + ComplementNB",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        ComplementNB(),
    )
)
results.append(
    evaluate(
        "TF-IDF (1,2) + RandomForest",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    )
)
# Способ 4: подбор гиперпараметров
param_grid = {
    "C": [0.1, 1, 10],
    "class_weight": [None, "balanced"],
}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring="f1_macro",
)
results.append(
    evaluate(
        "TF-IDF (1,2) + LogReg + GridSearch",
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
        grid,
    )
)
print(f"Лучшие параметры: {grid.best_params_}")

# Комбинация: лучший векторизатор + лучшая настроенная модель
grid_char = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring="f1_macro",
)
results.append(
    evaluate(
        "TF-IDF char_wb (2,5) + LogReg + GridSearch",
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=20000),
        grid_char,
    )
)
print(f"Лучшие параметры: {grid_char.best_params_}")

base_f1 = results[0]["f1"]

with open("reports/day4_results.md", "w", encoding="utf-8") as f:
    f.write("| Метод | Macro F1 | Δ к baseline | Δ, % |\n")
    f.write("|---|---|---|---|\n")
    for r in results:
        delta = r["f1"] - base_f1
        delta_pct = delta / base_f1 * 100
        f.write(f"| {r['name']} | {r['f1']:.4f} | {delta:+.4f} | {delta_pct:+.1f}% |\n")

best = max(results, key=lambda r: r["f1"])
print(f"\nЛучший вариант: {best['name']} - Macro F1 {best['f1']:.4f}")

print("\nBaseline по классам:")
print(
    classification_report(
        y_test, results[0]["pred"], target_names=list(LABEL2ID.keys())
    )
)
print("Лучший вариант по классам:")
print(classification_report(y_test, best["pred"], target_names=list(LABEL2ID.keys())))
