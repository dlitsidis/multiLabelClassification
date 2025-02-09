import joblib
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score
import pandas as pd
import tensorflow as tf
import keras
import time


df = pd.read_csv('Dataset/movies.csv', on_bad_lines='skip')
print(df.shape)
df = df[df['description'] != 'Add a Plot']
df = df[df['description'] != 'Plot under wraps.']
df = df[df['description'] != 'Plot kept under wraps.']
df = df[df['description'] != 'Plot unknown.']

df.drop_duplicates(inplace=True)
df.describe(include='all')

df.dropna(inplace=True)

if df.isnull().values.any():
    print("There are missing values in the dataset.")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    print("\nAfter dropping this rows")
    print(df.isnull().sum())
else:
    print("There are no missing values in the dataset.")

df.reset_index(drop=True, inplace=True)
df.describe(include='all')

# Assuming 'genre' is the column in your dataframe containing genres as strings
df['genre'] = df['genre'].apply(lambda x: [g.strip().lower() for g in x.split(',')] if isinstance(x, str) else [])

print(df.shape)
print(df.head())

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genre'])
genre_columns = mlb.classes_

genre_df = pd.DataFrame(genres_encoded, columns=genre_columns)
dataset = pd.concat([df, genre_df], axis=1)

dataset.drop('genre', axis=1, inplace=True)

print(dataset.shape)
dataset.head()

num_classes = mlb.classes_.shape[0]
print(num_classes)
# Step 1: Text Representation

descriptions = dataset['description']
tagged_descriptions = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(descriptions)]

# Train a doc2vec model from the start
doc2vec = Doc2Vec(vector_size=300, window=10, min_count=1, workers=2, epochs=50)
doc2vec.build_vocab(tagged_descriptions)
doc2vec.train(tagged_descriptions, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
document_vectors = [doc2vec.infer_vector( word_tokenize(doc.lower())) for doc in descriptions

X = document_vectors
X = np.array(X)
X = np.expand_dims(X, axis=1)
y = genres_encoded

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# encoder = keras.layers.TextVectorization(output_mode='tf-idf')
# encoder.adapt(X_train)
# # Use either simple LSTM or Bi-LSTM depending on your choice

# lstm = keras.layers.LSTM(128)
bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(128))

output_layer = keras.layers.Dense(num_classes, activation='sigmoid')
model = keras.Sequential([
    bi_lstm,  # Bi-LSTM layer
    keras.layers.Dropout(0.5),  # Add Dropout for regularization (optional)
    output_layer  # Final dense layer for predictions
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(len(X_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

start_time = time.time()
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
end_time = time.time()


y_pred = model.predict(X_test)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)
hamming_loss = hamming_loss(y_test, y_pred_binary)
f1_score = f1_score(y_test, y_pred_binary, average='samples')

test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")
print(f"Test f1_score: {f1_score}")
print(f"Test hamming_loss: {hamming_loss}")
print('Training time:', end_time - start_time)
