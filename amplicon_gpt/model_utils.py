import tensorflow as tf
from amplicon_gpt.losses import unifrac_loss_var, _pairwise_distances 
from amplicon_gpt.layers import SampleEncoder,  NucleotideEinsum, ReadHead

MAX_SEQ = 1600
BATCH_SIZE=8

def transfer_learn_base(batch_size: int, dropout: float):   
    d_model = 128
    dff = 128
    num_heads = 6
    num_enc_layers = 4
    norm_first = False
    
    input = tf.keras.Input(shape=[None,100], batch_size=batch_size, dtype=tf.int64)
    model_input = tf.keras.layers.Embedding(
        5,
        d_model,
        embeddings_initializer="glorot_uniform",
        input_length=100,
        input_shape=[batch_size, None, 100],
        name="embedding")(input)
    model_input = tf.keras.layers.LayerNormalization()(model_input)
    model_input = NucleotideEinsum(dff, input_max_length=100, normalize_output=True,  activation='relu')(model_input)
    model_input = NucleotideEinsum(128,
                               input_max_length=dff,
                               normalize_output=True,
                               activation='relu')(model_input)
    model_input = NucleotideEinsum(128,
                                   input_max_length=128,
                                   reduce_tensor=True,
                                   normalize_output=True,
                                   activation='relu')(model_input)
    model_input = SampleEncoder(dropout, num_enc_layers, num_heads, dff, norm_first)(model_input)
    output = ReadHead(d_model)(model_input)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

@tf.keras.saving.register_keras_serializable(package="amplicon_gpt.metrics")
class MAE(tf.keras.metrics.Metric):
    def __init__(self, name='mae_loss', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.loss = self.add_weight(name='rl', initializer='zero', dtype=tf.float32)
        self.i = self.add_weight(name='i', initializer='zero', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.loss.assign_add(tf.reduce_sum(tf.abs(_pairwise_distances(y_pred)-y_true)))
        self.i.assign_add(tf.constant(((BATCH_SIZE*BATCH_SIZE)-BATCH_SIZE) / 2.0, dtype=tf.float32))

    def result(self):
        return self.loss / self.i
    
def compile_model(model):
    # lr = 0.0001
    boundaries = [4400*4, 8800*4]
    values = [0.0001, 0.00005, 0.00001]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    # model.load_weights('base-model/encoder.keras')
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=unifrac_loss_var, metrics=[MAE()],
        jit_compile=False)
    return model
