import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# ----------------------
# Parameters
# ----------------------
vocab_size    = 1000      # maximum vocabulary size
embedding_dim = 16        # embedding output dimension
max_length    = 20        # max sequence length for padding/truncating
padding_type  = 'post'    # pad at the end
trunc_type    = 'post'    # truncate at the end
oov_tok       = '<OOV>'   # token for out-of-vocabulary words

# ----------------------
# 1. Basic Tokenization
# ----------------------
sentences_basic = [
    'I love my dog',
    'I love my cat'
]
tokenizer_basic = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer_basic.fit_on_texts(sentences_basic)
print('Basic word index:', tokenizer_basic.word_index)

# ----------------------
# 2. Sequencing Sentences
# ----------------------
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print('Word index:', tokenizer.word_index)
print('Sequences:', sequences)

# ----------------------
# 3. Test Data Sequencing
# ----------------------
test_data = [
    'I really love my dog',
    'My dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print('Test sequences:', test_seq)

# ----------------------
# 4. Padding Sequences
# ----------------------
padded = pad_sequences(sequences, maxlen=max_length,
                       padding=padding_type, truncating=trunc_type)
print('Padded sequences:\n', padded)

# ----------------------
# 5. Load Sarcasm Dataset
# ----------------------
with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences_sarcasm = [item['headline'] for item in datastore]
labels_sarcasm    = [item['is_sarcastic'] for item in datastore]

# ----------------------
# 6. Tokenize & Pad Sarcasm Data
# ----------------------
tokenizer_sarcasm = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer_sarcasm.fit_on_texts(sentences_sarcasm)
sequences_sarcasm = tokenizer_sarcasm.texts_to_sequences(sentences_sarcasm)
padded_sarcasm = pad_sequences(sequences_sarcasm, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)
print('Sarcasm data shape:', padded_sarcasm.shape)

# ----------------------
# 7. Define & Compile Sentiment Model
# ----------------------
sentiment_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
sentiment_model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

# Example training call (uncomment and provide data splits):
# history = sentiment_model.fit(
#     training_padded, training_labels,
#     epochs=30,
#     validation_data=(testing_padded, testing_labels),
#     verbose=2
# )

# ----------------------
# 8. RNN/LSTM Model Example
# ----------------------
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Example training call for RNN (uncomment and provide data):
# history_rnn = rnn_model.fit(
#     training_padded, training_labels,
#     epochs=10,
#     validation_data=(testing_padded, testing_labels)
# )

# ----------------------
# 9. Poetry Generation Example
# ----------------------
data = """In the town of Athy one Jeremy Lanigan
Battered away 'till he hadn't a pound."""
corpus = data.lower().split("\n")

tokenizer_poetry = Tokenizer()
tokenizer_poetry.fit_on_texts(corpus)
total_words = len(tokenizer_poetry.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer_poetry.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences and split into features/labels
max_seq_len = max(len(seq) for seq in input_sequences)
input_padded = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

xs = input_padded[:, :-1]
labels = input_padded[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

poetry_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 240, input_length=max_seq_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])
poetry_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     metrics=['accuracy'])

# Example poetry model training (uncomment to run):
# history_poetry = poetry_model.fit(xs, ys, epochs=100, verbose=1)

# Generate new poetry
seed_text = "I made a poetry machine"
next_words = 20
for _ in range(next_words):
    token_list = tokenizer_poetry.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted_probs = poetry_model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted_probs, axis=-1)[0]
    for word, index in tokenizer_poetry.word_index.items():
        if index == predicted_id:
            seed_text += ' ' + word
            break
print(seed_text)
