import tensorflow as tf


@tf.keras.saving.register_keras_serializable(
    package="sepsis.layer"
)
class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 emb_vocab,
                 emb_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb_vocab = emb_vocab
        self.emb_dim = emb_dim
        self.feature_tokens = tf.keras.layers.StringLookup(
            vocabulary=emb_vocab,
            mask_token='<MASK>',
            num_oov_indices=0,
            output_mode='int')
        self.feature_embedding = tf.keras.layers.Embedding(
            input_dim=len(emb_vocab)+1,
            output_dim=emb_dim,
            embeddings_initializer="uniform")
        
    def call(self, inputs):
        feature, rclr = inputs
        feature_embedding = self.feature_tokens(feature)
        feature_embedding = self.feature_embedding(feature_embedding)
        feature_embedding, _ = tf.linalg.normalize(feature_embedding, axis=-1)

        if not tf.is_symbolic_tensor(rclr):
            feature_embedding = tf.math.multiply(feature_embedding, rclr)
        
        return feature_embedding
