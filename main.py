import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
