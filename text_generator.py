import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download the Shakespeare dataset
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

# Read data
# read file in binary and decode it into unicode string
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print(f"Length of text: {len(text)} characters")

# Set vocabulary size and maximum sequence length
vocab_size = 20000
max_length = 200

# Initialize tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>', filters='')

# Tokenize the text
tokenizer.fit_on_texts([text])

# Prepare sequences for text generation
sequences = tokenizer.texts_to_sequences([text])
# convert sequences into a numpy array and reshape it
sequences = np.array(sequences).reshape(-1)
# input sequences (no last one)
X_train = sequences[:-1]
# target/output sequences (no first one)
y_train = sequences[1:]

# predict next word given previous word
# model is trained to minimize the differenc predictions and y_train

# build the model
model = Sequential([
    Embedding(vocab_size, 32, input_length=1), #embedding layer
    LSTM(100), #long short-term memory layer
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Save the trained model
model.save('text_generator/text_generation_model.h5') 

# Load the saved model
loaded_model = tf.keras.models.load_model('text_generator/text_generation_model.h5')

# Function to generate text using the loaded model
def generate_text(seed_text, max_length, loaded_model, tokenizer, temperature=1.0):
    loaded_model.reset_states()  # Reset states when starting a new sequence
    generated_text = seed_text
    for _ in range(max_length):
        seed_sequence = tokenizer.texts_to_sequences([seed_text])
        padded_sequence = pad_sequences(seed_sequence, maxlen=1)
        predicted_probs = loaded_model.predict(padded_sequence)[0]
        
        # Apply temperature to control randomness
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)
        
        # Sample the next word
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        predicted_word = tokenizer.index_word.get(predicted_index, '<OOV>')
        
        # Replace <OOV> with a random word from the vocabulary
        if predicted_word == '<OOV>':
            predicted_word = np.random.choice(list(tokenizer.word_index.keys()))
        
        seed_text += ' ' + predicted_word
        generated_text += ' ' + predicted_word
    return generated_text

# Generate new text
seed_text = "ROMEO:"
generated_text = generate_text(seed_text, max_length, loaded_model, tokenizer, temperature=0.5)
print("Generated Text:\n", generated_text)
