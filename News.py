import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Lambda, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

# Parameters
embedding_dim = 256
gru_units = 512
vocab_size = 10000  # Adjust based on your dataset
max_sequence_length = 500  # Adjust according to article length

# Encoder Model
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
encoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='encoder_gru')
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)

# Decoder Model
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
decoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='decoder_gru')
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)

# Dot Product Attention
attention_scores = Dot(axes=[2, 2], name='attention_scores')([decoder_outputs, encoder_outputs])
attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
context_vector = Dot(axes=[2, 1], name='context_vector')([attention_weights, encoder_outputs])

# Concatenate context vector with decoder outputs
concat_output = Concatenate(name='concat_layer')([decoder_outputs, context_vector])

# Dense layer for word prediction
decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(concat_output)

# Final Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
print(model.summary())
