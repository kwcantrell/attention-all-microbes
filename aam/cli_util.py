import click


def _aam_globals():
    return {
        'feature-attention-methods': ['add_features', 'mask_features', 'none']
    }


aam_globals = _aam_globals()


def aam_model_options(func):
    model_options = [
        click.option(
            '--p-batch-size',
            default=32,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-epochs',
            default=1000,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-repeat',
            default=5,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-dropout',
            default=0.1,
            show_default=True,
            type=float
        ),
        click.option(
            '--p-token-dim',
            default=512,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-feature-attention-method',
            default='add_features',
            type=click.Choice(aam_globals['feature-attention-methods'])
        ),
        click.option(
            '--p-features-to-add-rate',
            default=1.,
            show_default=True,
            type=float
        ),
        click.option(
            '--p-ff-d-model',
            default=128,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-ff-clr',
            default=1024,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-pca-heads',
            default=4,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-enc-layers',
            default=2,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-enc-heads',
            default=8,
            show_default=True,
            type=int
        ),
        click.option(
            '--p-lr',
            default=0.001,
            show_default=True,
            type=float
        ),
        click.option(
            '--p-report-back-after',
            default=5,
            show_default=True,
            type=int
        ),
    ]

    for option in reversed(model_options):
        func = option(func)
    return func
