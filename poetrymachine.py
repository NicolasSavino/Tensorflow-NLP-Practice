import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# ----------------------
# 1. Prepare the corpus
# ----------------------
data = """In the town of Athy one Jeremy Lanigan
Battered away 'till he hadn't a pound."""
corpus = data.lower().split("\n")

# ----------------------
# 2. Tokenize
# ----------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# ----------------------
# 3. Create n-gram sequences
# ----------------------
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):            # use `i`, not `1`
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

# ----------------------
# 4. Pad & split into features/labels
# ----------------------
max_seq_len = max(len(seq) for seq in input_sequences)
padded_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

xs = padded_sequences[:, :-1]
labels = padded_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# ----------------------
# 5. Build the model
# ----------------------
model = Sequential([
    Embedding(input_dim=total_words, output_dim=240, input_length=max_seq_len - 1),
    Bidirectional(LSTM(150)),
    Dense(total_words, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# ----------------------
# 6. Train (uncomment to run)
# ----------------------
# history = model.fit(xs, ys, epochs=100, verbose=1)

# ----------------------
# 7. Generate new text
# ----------------------
seed_text = "I made a poetry machine"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted_probs, axis=-1)[0]
    
    # find the word corresponding to the predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            seed_text += ' ' + word
            break

print(seed_text)
