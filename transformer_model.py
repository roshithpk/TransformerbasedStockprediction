import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_model(input_shape, num_heads=4, ff_dim=128, dropout=0.2):
    """
    Builds a Transformer-based model for time series forecasting.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_features)
        num_heads (int): Number of attention heads
        ff_dim (int): Hidden layer size in feed-forward network
        dropout (float): Dropout rate

    Returns:
        tf.keras.Model: Compiled Transformer model
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding Layer
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    # Transformer Block
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
    attention_output = layers.Dropout(dropout)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(input_shape[1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    return model


class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_len, d_model)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )

        angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000., (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return tf.cast(pos, tf.float32) * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

