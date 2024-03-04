import click


def aam_model_options(func):
    model_options = [
        click.option('--p-batch-size',
                     default=8,
                     type=int),
        click.option('--p-train-percent',
                     default=0.8,
                     type=float),
        click.option('--p-epochs',
                     default=100,
                     type=int),
        click.option('--p-repeat',
                     default=1,
                     type=int),
        click.option('--p-d-model',
                     default=64,
                     type=int),
        click.option('--p-pca-hidden-dim',
                     default=64,
                     type=int),
        click.option('--p-pca-heads',
                     default=4,
                     type=int),
        click.option('--p-t-heads',
                     default=4,
                     type=int),
        click.option('--p-output-dim',
                     default=1,
                     type=int),
    ]

    for option in reversed(model_options):
        func = option(func)
    return func
