import tensorflow as tf
import tensorflow_models as tfm

from aam.losses import PairwiseLoss
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.transformer_decoder import TransformerDecoder


class GOTUModel(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        token_limit: int,
        encoder_type: str = "unifrac",
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "gelu",
        max_bp: int = 150,
        is_16S: bool = True,
        vocab_size: int = 6,
        add_token: bool = True,
        asv_dropout_rate: float = 0.0,
        accumulation_steps: int = 1,
        nucleotide_encoder=None,
        pairwise_loss_type="mse",
        gotu_count=None,
        max_gotu=1024,
        base_model=None,
        freeze_base_weights=False,
        bert_training=False,
        **kwargs,
    ):
        super(GOTUModel, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.token_limit = token_limit
        self.encoder_type = encoder_type
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.max_bp = max_bp
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.add_token = add_token
        self.asv_dropout_rate = asv_dropout_rate
        self.accumulation_steps = accumulation_steps
        self.nucleotide_encoder = nucleotide_encoder
        self.pairwise_loss_type = pairwise_loss_type
        self.gotu_count = gotu_count
        self.max_gotu = max_gotu
        self.base_model = base_model
        self.freeze_base_weights = freeze_base_weights
        self.bert_training = bert_training

        self.encoder_tracker = tf.keras.metrics.Mean()
        self.loss_tracker = tf.keras.metrics.Mean()
        self.gotu_tracker = tf.keras.metrics.Mean()
        self.nuc_tracker = tf.keras.metrics.Mean()

        self.nuc_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.encoder_loss = PairwiseLoss(self.pairwise_loss_type, reduction="none")
        self.gotu_loss = tf.keras.losses.CategoricalCrossentropy(reduction="none")

        self.gotu_embedding_layer = tf.keras.layers.Embedding(
            self.gotu_count + 2,
            self.embedding_dim,
        )

        self.gotu_pos_emb = tfm.nlp.layers.PositionEmbedding(
            self.max_gotu, seq_axis=1, initializer="zeros"
        )
        if self.base_model is not None:
            self.asv_embedding_layer = self.base_model
            if freeze_base_weights is True:
                self.asv_embedding_layer.trainable = False
        else:
            self.asv_embedding_layer = SequenceEncoder(
                output_dim=self.output_dim,
                token_limit=self.token_limit,
                encoder_type=self.encoder_type,
                dropout_rate=self.dropout_rate,
                embedding_dim=self.embedding_dim,
                attention_heads=self.attention_heads,
                attention_layers=self.attention_layers,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.intermediate_activation,
                max_bp=self.max_bp,
                is_16S=self.is_16S,
                vocab_size=self.vocab_size,
                add_token=self.add_token,
                asv_dropout_rate=self.asv_dropout_rate,
                accumulation_steps=self.accumulation_steps,
                nucleotide_encoder=self.nucleotide_encoder,
                pairwise_loss_type=self.pairwise_loss_type,
            )
        self.gotu_decoder = TransformerDecoder(
            num_attention_heads=self.attention_heads,
            num_layers=self.attention_layers,
            intermediate_size=self.intermediate_size,
            activation=self.intermediate_activation,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.dropout_rate,
        )
        self._softmax = tf.keras.layers.Activation("softmax", dtype=tf.float32)
        self.gotu_output = tf.keras.layers.Dense(self.gotu_count + 2)

    def get_config(self):
        config = super(SequenceEncoder, self).get_config()
        base_model = None
        if self.base_model is not None:
            base_model = tf.keras.saving.serialize_keras_object(self.asv_embedding_layer)
        config.update(
            {
                "output_dim": self.output_dim,
                "token_limit": self.token_limit,
                "encoder_type": self.encoder_type,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "max_bp": self.max_bp,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "add_token": self.add_token,
                "asv_dropout_rate": self.asv_dropout_rate,
                "accumulation_steps": self.accumulation_steps,
                "gotu_count": self.gotu_count,
                "max_gotu": self.max_gotu,
                "base_model": base_model,
                "freeze_base_weights": self.freeze_base_weights,
                "bert_training": self.bert_training,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        base_model = config['base_model']
        if base_model is not None:
            config["base_model"] = tf.keras.saving.deserialize_keras_object(base_model)
        model = cls(**config)
        return model
        
    def _compute_unifrac_loss(
        self,
        y_true: tf.Tensor,
        unifrac_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        batch_size = tf.shape(y_true)[0]
        pairs = tf.linalg.band_part(
            tf.ones((batch_size, batch_size), dtype=tf.float32), 0, -1
        )
        pairs = tf.reduce_sum(pairs)

        loss = self._unifrac_loss(y_true, unifrac_embeddings)
        loss = tf.reduce_sum(loss / pairs)
        return loss / 2

    def _compute_loss(self, asv_inputs, gotu_inputs, outputs):
        asv_x, asv_y = asv_inputs
        asv_tokens, asv_counts = asv_x
        asv_target, asv_unifrac_dist = asv_y
        gotu_x, gotu_y = gotu_inputs
        gotu_tokens, gotu_counts = gotu_x
        gotu_pred, unifrac_pred, nuc_mask, nuc_pred = outputs

        gotu_tokens = tf.pad(gotu_tokens, [[0, 0], [0, 1]], constant_values=2)

        asv_tokens = (
            asv_tokens
            + self.asv_embedding_layer.base_encoder.asv_encoder.nucleotide_position
        )
        asv_tokens = tf.reshape(asv_tokens, shape=[-1])
        nuc_mask = tf.reshape(nuc_mask, shape=[-1])
        asv_tokens = asv_tokens[nuc_mask]
        asv_tokens = tf.one_hot(asv_tokens, tf.shape(nuc_pred)[-1])

        nuc_loss = self.nuc_loss(asv_tokens, nuc_pred)
        nuc_loss = tf.reduce_mean(nuc_loss)

        unifrac_loss = self._compute_unifrac_loss(asv_unifrac_dist, unifrac_pred)

        gotu_tokens = tf.one_hot(gotu_tokens, tf.shape(gotu_pred)[-1])
        gotu_mask = tf.reshape(gotu_counts > 1, shape=[-1])
        gotu_tokens = tf.reshape(gotu_tokens, shape=[-1, tf.shape(gotu_pred)[-1]])[
            gotu_mask
        ]
        gotu_pred = tf.reshape(gotu_pred, shape=[-1, tf.shape(gotu_pred)[-1]])[
            gotu_mask
        ]
        gotu_loss = self.gotu_loss(gotu_tokens, gotu_pred)
        gotu_loss = tf.reduce_mean(gotu_loss)

        loss = gotu_loss + unifrac_loss + nuc_loss

        return loss, gotu_loss, nuc_loss, unifrac_loss

    def train_step(self, data):
        asv_inputs, gotu_inputs = data
        asv_x, asv_y = asv_inputs
        gotu_x, gotu_y = gotu_inputs
        with tf.GradientTape() as tape:
            outputs = self((asv_x, gotu_x), training=True)
            loss, gotu_loss, nuc_loss, encoder_loss = self._compute_loss(
                asv_inputs, gotu_inputs, outputs
            )
            if self.compute_dtype == "float16":
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.compute_dtype == "float16":
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(loss)
        self.encoder_tracker.update_state(encoder_loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "encoder_loss": self.encoder_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def test_step(self, data):
        asv_inputs, gotu_inputs = data
        asv_x, asv_y = asv_inputs
        gotu_x, gotu_y = gotu_inputs
        outputs = self((asv_x, gotu_x), training=True)
        loss, gotu_loss, nuc_loss, encoder_loss = self._compute_loss(
            asv_inputs, gotu_inputs, outputs
        )

        self.loss_tracker.update_state(loss)
        self.gotu_tracker.update_state(loss)
        self.encoder_tracker.update_state(encoder_loss)
        self.nuc_tracker.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "gotu_loss": self.gotu_tracker.result(),
            "encoder_loss": self.encoder_tracker.result(),
            "nuc_loss": self.nuc_tracker.result(),
            "learning_rate": self.optimizer.learning_rate,
        }

    def call(
        self,
        asv_inputs=tuple[tf.Tensor, tf.Tensor],
        gotu_inputs=tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        asv_tokens, asv_counts = asv_inputs
        gotu_tokens, gotu_counts = gotu_inputs
        # Refers to dict in GOTU Generator
        # TODO: change padding to happen in data generator
        gotu_tokens = tf.pad(gotu_tokens, [[0, 0], [1, 0]], constant_values=1)
        asv_mask = asv_counts > 0
        gotu_mask = gotu_counts > 0

        gotu_embeddings = self.gotu_embedding_layer(gotu_tokens)
        gotu_embeddings = gotu_embeddings + self.gotu_pos_emb(gotu_embeddings)

        asv_embeddings, unifrac_pred, nuc_mask, nuc_pred = self.asv_embedding_layer(
            asv_tokens,
            include_bert_random_mask=self.bert_training,
            training=training,
        )
        gotu_pred = self.gotu_decoder(
            asv_embeddings, gotu_embeddings, asv_mask, gotu_mask, training=training
        )
        gotu_pred = self.gotu_output(gotu_pred)
        gotu_pred = self._softmax(gotu_pred)

        return gotu_pred, unifrac_pred, nuc_mask, nuc_pred
