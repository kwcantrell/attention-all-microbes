import tensorflow as tf

from aam.common.callbacks import SaveModel


def train(
    model: tf.keras.Model,
    reg_out_callbacks: list,
    lr: float,
    output_dir: str,
    report_back_after: int,
    training_dataset,
    validation_dataset,
    epochs: int,
):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    model.compile(
        optimizer=optimizer,
        jit_compile=False,
    )
    core_callbacks = [
        # tboard_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            "loss", factor=0.5, patients=2, min_lr=0.000001
        ),
        tf.keras.callbacks.EarlyStopping("loss", patience=50),
        SaveModel(output_dir, report_back_after),
    ]
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        callbacks=[*reg_out_callbacks, *core_callbacks],
        epochs=epochs,
    )
    return model.summary()


# def pretrain_unifrac(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)

#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
#     )
#     model.compile(
#         optimizer=optimizer,
#         # loss=PairwiseLoss(),
#         # metrics=[PairwiseMAE()],
#         jit_compile=False,
#     )
#     return model


# def transfer_regression(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     model.build(input_shape=[8, None, 100])
#     model.load_weights("base-model-large/encoder.keras")
#     model.trainable = False
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
#     )
#     output = tfm.nlp.models.TransformerEncoder(
#         num_layers=4,
#         num_attention_heads=8,
#         intermediate_size=1024,
#         dropout_rate=0.1,
#         norm_first=True,
#         activation="relu",
#     )(model.layers[-2].output)
#     output = ReadHead(hidden_dim=128, num_heads=8, num_layers=1, output_dim=1)(output)
#     model = tf.keras.Model(inputs=model.layers[0].input, outputs=output)
#     loss = pairwise_residual_mse(batch_size=batch_size)
#     model.compile(
#         optimizer=optimizer,
#         loss=pairwise_residual_mse(batch_size),
#         metrics=["mae"],
#         jit_compile=False,
#     )
#     return model


# def classification(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.999, epsilon=1e-7)
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     model.compile(
#         optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=False
#     )
#     return model


# def regression(batch_size: int, lr: float, *args):
#     model = _construct_base(batch_size, *args)
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, beta_2=0.999, epsilon=1e-7)
#     loss = pairwise_residual_mse(batch_size=batch_size)
#     model.compile(optimizer=optimizer, loss="mse", metrics=["mae"], jit_compile=False)
#     return model
