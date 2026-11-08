import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

with open('reports/day_1_observations.md', 'wb') as file:
    file.write()