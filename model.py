
# model.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, model_dim):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_len, model_dim)

    def get_config(self):
        return {"sequence_len": self.pos_encoding.shape[1], "model_dim": self.pos_encoding.shape[2]}

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def build_transformer_model(seq_len, n_features, d_model=64, n_heads=4, ff_dim=128):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = PositionalEncoding(seq_len, d_model)(inputs)

    attention_output = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(d_model)
    ])
    x_ffn = ffn(x)
    x = layers.Add()([x, x_ffn])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
