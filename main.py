import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocess import preprocess_text, LABEL2ID


df = pd.read_csv("data.csv")
df = df.rename(columns={"Sentence": "text", "Sentiment": "sentiment"})

print(f"Размерность - {df.shape}")
print(f"Первые 5 строк:\n{df.head()}")
df.info()
print(f"Количество пропущенных значений:\n{df.isnull().sum()}")

print(df['sentiment'].value_counts())
print(df['sentiment'].value_counts(normalize=True))

plt.figure()
sns.countplot(data=df, x="sentiment")
plt.title("Classes spread")
plt.savefig("reports/sentiment_distribution.png")
plt.close()

df['text_length'] = df["text"].str.len()
df['word_count'] = df["text"].str.split().str.len()
print(df[['text_length', 'word_count']].describe())
print(df.groupby('sentiment')['word_count'].mean())

plt.figure()
sns.histplot(data=df, x='text_length', bins=50)
plt.title('Len histogram')
plt.xlabel('Text length')
plt.ylabel('Text count')
plt.savefig("reports/text_length_distribution.png")
plt.close()

for sentiment in df['sentiment'].unique():
    print(f"\n=== Класс {sentiment} ===")
    for example in df[df['sentiment'] == sentiment]['text'].sample(3, random_state=42):
        print(f'Example - {example}')


df['text_clean'] = df['text'].apply(preprocess_text)

rows_before = len(df)
df = df[df['text_clean'].str.len() > 0]
print(f"Удалено пустых после очистки: {rows_before - len(df)}")

rows_before = len(df)
df = df.drop_duplicates(subset=['text_clean', 'sentiment'])
print(f"Удалено полных дубликатов: {rows_before - len(df)}")

print(f"Осталось строк: {len(df)}")


df['label'] = df['sentiment'].map(LABEL2ID)
print(f"Неизвестных меток: {df['label'].isnull().sum()}")
print(df[['sentiment', 'label']].value_counts())


checks = {
    'ссылка': df['text'].str.contains('http'),
    'кавычки': df['text'].str.contains('``', regex=False),
    'пробел перед знаком': df['text'].str.contains(' , ', regex=False),
    "отдельное 's": df['text'].str.contains(" 's", regex=False)
}
for name, mask in checks.items():
    print(f"\n=== {name} ===")
    for _, row in df[mask].head(2).iterrows():
        print(f"ДО:     {row['text']}")
        print(f"ПОСЛЕ:  {row['text_clean']}")
        print()


train_df, test_df = train_test_split(
    df, test_size=.2, random_state=42, stratify=df['label']
)
print(f"\nTrain: {len(train_df)} строк, Test: {len(test_df)} строк")
print("Доли классов в train:")
print(train_df['label'].value_counts(normalize=True).sort_index())
print("Доли классов в test:")
print(test_df['label'].value_counts(normalize=True).sort_index())


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text_clean'])
X_test = vectorizer.transform(test_df['text_clean'])
y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
print(f"Пример признако: {vectorizer.get_feature_names_out()[:15]}")


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nПервые 10 предсказаний: {y_pred[:10]}")
print(f"\nПервые 10 правильных: {y_test[:10]}")

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(classification_report(y_test, y_pred, target_names=list(LABEL2ID.keys())))


baseline_results = {
    'model': 'TF-IDF (1,2), max_features=5000 + LogisticRegression',
    'accuracy': round(accuracy, 4),
    'macro_f1': round(macro_f1, 4),
    'train_size': len(train_df),
    'test_size': len(test_df),
}

with open('reports/baseline_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)

with open('reports/baseline_report.txt', 'w', encoding='utf-8') as f:
    f.write(classification_report(y_test, y_pred, target_names=list(LABEL2ID.keys())))

print("Baseline результаты сохранены в reports/")

