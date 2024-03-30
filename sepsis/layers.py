import tensorflow as tf


class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 token_dim,
                 emb_vocab,
                 emb_dim,
                 dropout,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb_vocab = emb_vocab
        tokens = tf.constant([i for i in range(len(emb_vocab))], dtype=tf.int64)
        self.tokens = tf.expand_dims(tokens, axis=0)
        self.emb_dim = emb_dim
        self.feature_tokens = tf.keras.layers.StringLookup(
            vocabulary=emb_vocab,
            mask_token='<MASK>',
            num_oov_indices=0,
            output_mode='int')
        self.feature_embedding = tf.keras.layers.Embedding(
            input_dim=len(emb_vocab)+1,
            output_dim=token_dim,
            embeddings_initializer="ones",
            dtype=tf.float32)
        self.dff = tf.keras.layers.Dense(emb_dim, dtype=tf.float32)
        self.flag = self.add_weight(trainable=False)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        feature, rclr = inputs
        feature_tokens = self.feature_tokens(feature)
        
        if self.flag == 0:
            self.flag.assign_add(1)
            feature_embedding = self.feature_embedding(feature_tokens)
            output = feature_embedding
        else:
            batch_size = tf.shape(feature_tokens)[0]
            feature_size = tf.shape(feature_tokens)[1]
            tokens = tf.repeat(self.tokens, [batch_size], axis=0)
            tokens = tf.random.shuffle(tokens)
            tokens = tokens[:, :tf.shape(feature_tokens)[1]]
            mask = (1 - tf.sequence_mask([5], feature_size, dtype=tf.int64))
            f_mask = tf.equal(feature_tokens, tf.constant(0, dtype=tf.int64))
            mask *= tf.cast(f_mask, dtype=tf.int64)
            tokens *= tf.cast(mask, dtype=tf.int64)
            
            feature_tokens += tokens
            rclr += tf.cast(mask, dtype=tf.float32)
            feature_embedding = self.feature_embedding(feature_tokens)
            output = feature_embedding * tf.expand_dims(rclr, axis=-1)
            output = self.layer_norm(output)
            output = self.dropout(output, training)
        output = self.dff(output)
        return output


class PCA(tf.keras.layers.Layer):
    def __init__(self,
                 emb_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.flag = self.add_weight(trainable=False)

    def call(self, inputs):
        if self.flag == 0:
            self.flag.assign_add(1)
            output = inputs
        else:
            output = self.pca(inputs)
        return output
    
    def pca(self, input):
        output = input - tf.math.reduce_mean(input,
                                              axis=-2,
                                              keepdims=True)
        cov = tf.linalg.matmul(output, output, transpose_a=True)
        _, output = tf.linalg.eigh(cov)
        output = tf.transpose(output, perm=[0,2,1])
        return output

class ProjectDown(tf.keras.layers.Layer):
    def __init__(self,
                 emb_dim,
                 dims,
                 reduce_dim,
                 **kwargs):
        super().__init__(**kwargs)
        
        if dims == 3:
            shape = [None, emb_dim, emb_dim]
        else:
            shape = [None, emb_dim]

        self.proj_down = tf.keras.layers.Dense(1)
        self.proj_down.build(shape)
        self.reduce_dim = reduce_dim

    def call(self, inputs):
        if self.reduce_dim:
            output = tf.squeeze(self.proj_down(inputs), axis=-1)
        else:
            output = self.proj_down(inputs)
        return output
