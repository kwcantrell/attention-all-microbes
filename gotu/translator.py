import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table
from unifrac import unweighted


class Translator(tf.Module):
    def init(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def call(self, sentence, max_length=MAX_TOKENS):
        # The input sentence is Portuguese, hence adding the [START] and [END] tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English [START] token.
        start_end = self.tokenizers.en.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # tf.TensorArray is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by tf.function.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the seq_len dimension.
            predictions = predictions[:, -1:, :]  # Shape (batch_size, 1, vocab_size).

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the predicted_id to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is (1, tokens).
        text = tokenizers.en.detokenize(output)[0]  # Shape: ().

        tokens = tokenizers.en.lookup(output)[0]

        # tf.function prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights
