# Text-Generator-Shakespeare
Tutorial: https://github.com/GoogleCloudPlatform/asl-ml-immersion/blob/master/notebooks/text_models/solutions/text_generation.ipynb

This code implements a text generation model using a recurrent neural network (RNN) with LSTM layers

## Prepare Data
The data is downloaded and read from a Shakespeare dataset.

## Tokenization
Initializes a Tokenizer to convert text into numerical sequences.

Fits the tokenizer on the text data and converts the text into sequences.

## Prepare Sequences
Reshapes the sequences into a numpy array.

Creates two sets of sequences, X_train and y_train:
  - X_train: Input sequences, excluding the last one.
  - y_train: Target/output sequences, excluding the first one.

## Build the Model
Defines a sequential model with three layers:
  - Embedding layer: Maps each word to a vector of size 32.
  - LSTM layer: Long Short-Term Memory layer with 100 units.
  - Dense layer: Fully connected layer with softmax activation for predicting the next word.

## Compile the Model
Configures the model for training using the Adam optimizer and sparse categorical crossentropy loss

## Train the Model
Fits the model on the training data (X_train and y_train) for 10 epochs. 
(Note: you can increase the epochs)

## Save the Model
Saves the trained model as 'text_generation_model.h5'.

## Load the Model
Loads the saved model using tf.keras.models.load_model.

## Generate Text
Defines a function generate_text that takes a seed text and generates new text using the loaded model.

The generated text is influenced by a temperature parameter, controlling the randomness of the predictions.

Calls the generate_text function with a seed text "ROMEO:" and prints the generated text.
