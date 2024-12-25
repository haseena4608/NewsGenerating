from sklearn.model_selection import train_test_split

# Sample data for demonstration (replace with your full dataset)
articles = ["Sample text for the encoder input.", "Another example of encoder input."]
headlines = ["Generated headline for decoder input.", "Another headline example."]

# Split the data into training and test sets
articles_train, articles_test, headlines_train, headlines_test = train_test_split(articles, headlines, test_size=0.2, random_state=42)

# Tokenize and convert text to sequences
encoder_input_sequences_test = tokenizer.texts_to_sequences(articles_test)
decoder_input_sequences_test = tokenizer.texts_to_sequences(headlines_test)

# Pad sequences to ensure uniform input size
X_test_encoder = pad_sequences(encoder_input_sequences_test, maxlen=500, padding='post')
X_test_decoder = pad_sequences(decoder_input_sequences_test, maxlen=500, padding='post')

# Create target sequences for testing (shifted decoder sequences)
y_test = X_test_decoder[:, 1:]
y_test = pad_sequences(y_test, maxlen=500, padding='post')

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test_encoder, X_test_decoder], y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
