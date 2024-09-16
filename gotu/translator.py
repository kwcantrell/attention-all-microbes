import numpy as np
import pandas as pd
import tensorflow as tf
from biom import load_table
from unifrac import unweighted
from aam.utils import float_mask


def pre_sort(tensor, start_token):
    tensor = tf.squeeze(tensor, axis=0)
    valid_tokens = tensor[1:]
    sorted_indices = tf.argsort(valid_tokens)
    valid_tokens = tf.gather(valid_tokens, sorted_indices)
    valid_tokens = tf.expand_dims(valid_tokens, axis=0)
    return tf.pad(valid_tokens, [[0, 0], [1, 0]], constant_values=start_token)


class Translator(tf.Module):
    def __init__(self, transformer, start_token, end_token):
        self.transformer = transformer
        self.start_token = start_token
        self.end_token = end_token

    def __call__(self, sentence):
        encoder_input = sentence
        start = tf.reshape(
            tf.constant(self.start_token, dtype=tf.int64),
            shape=[
                1,
            ],
        )
        end = tf.constant(self.end_token, dtype=tf.int64)

        # tf.TensorArray is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by tf.function.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        # output_array = output_array.write(1, [558])
        # tf.print("ENCODER INPUTS SHAPE", encoder_input.indices.shape)
        encoder_inputs = tf.gather(self.transformer.o_ids, encoder_input.indices)

        encoder_inputs = tf.cast(
            self.transformer.sequence_tokenizer(encoder_inputs), dtype=tf.int32
        )
        #  tf.print("ENCODER INPUTS SHAPE AFTER", encoder_inputs.shape)

        encoder_inputs = tf.expand_dims(encoder_inputs, axis=0)

        encoder_output = self.transformer.base_model.feature_emb(
            (encoder_inputs, encoder_input.values), return_nuc_attention=False
        )
        encoder_mask = tf.reduce_sum(encoder_inputs, axis=-1, keepdims=True)
        encoder_mask = tf.pad(
            encoder_mask, paddings=[[0, 0], [0, 1], [0, 0]], constant_values=1
        )
        encoder_mask = float_mask(encoder_mask)

        for i in tf.range(self.transformer.num_gotus):
        # for i in tf.range(1000):
            output = tf.transpose(output_array.stack())
            # tf.print("OUTPUT", tf.shape(output), output)
            predictions = self.transformer.infer([encoder_output, encoder_mask, output])
            # Select the last token from the seq_len dimension.
            # predictions = predictions[:, -1:, :]  # Shape (batch_size, 1, vocab_size).
            # tf.print("PREDICTIONS", tf.shape(predictions))
            predicted_id = tf.argmax(predictions, axis=-1)
            # tf.print("PREDICTED_ID", predicted_id)
            # tf.print("PRED_ID", predicted_id)

            # Concatenate the predicted_id to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id)
            predicted_id = tf.squeeze(predicted_id)
            if predicted_id == end:
                tf.print("FORBIDDEN IF STATEMENT", predicted_id, i)
                break
            # tf.print("ITERATION OF PAIN #", i, predicted_id, end)
        output = tf.transpose(output_array.stack())
        output = tf.squeeze(output, axis=0)
        tf.print("OUTPUT", output, tf.shape(output))
        # tf.print("OUTPUT", tf.reduce_sum(output, axis=-1), tf.shape(output))
        output_size = tf.size(output)
        output = tf.pad(output, [[0, 1002 - output_size]])

        # tf.print("OUTPUT", output)
        # # The output shape is (1, tokens).
        # # text = self.tokenizer.detokenize(output)[0]  # Shape: ().

        # # tokens = self.tokenizer.lookup(output)[0]

        # # tf.function prevents us from using the attention_weights that were
        # # calculated on the last iteration of the loop.
        # # So, recalculate them outside the loop.
        # # self.transformer([encoder_input, output[:, :-1]], training=False)
        # # attention_weights = self.transformer.decoder.last_attn_scores

        return output
        # return tf.constant([420, 69, 1001], dtype=tf.int64)
