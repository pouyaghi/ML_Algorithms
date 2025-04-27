#Path to dataset files: /home/pouya/.cache/kagglehub/datasets/ronikdedhia/next-word-prediction/versions/1
from sklearn.preprocessing import LabelEncoder
import re

# Example

with open("/home/pouya/.cache/kagglehub/datasets/ronikdedhia/next-word-prediction/versions/1/1661-0.txt", 'r', encoding='utf-8') as file:
    text = file.read()

#print(text[:501])  # Print first 500 characters


text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()
words = words[:10000]

sequence_length = 3
X, y = [], []

for i in range(sequence_length, len(words)):
    X.append(words[i-sequence_length:i])
    y.append(words[i])

sequence_length = 3
X, y = [], []

for i in range(sequence_length, len(words)):
    X.append(words[i-sequence_length:i])
    y.append(words[i])

encoder = LabelEncoder()
all_words = list(set(words))
encoder.fit(all_words)

# Encode your input and target
X_encoded = [encoder.transform(seq) for seq in X]
y_encoded = encoder.transform(y)

def get_data():
    return X_encoded, y_encoded, encoder