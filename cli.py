import click
import biom


TABLE_DESC = (
    "Feature table containing all features that should be used for "
    "target prediction."
)
METADATA_COL_DESC = (
    "Categorical metadata column to use as prediction target."
)
MISSING_SAMPLES_DESC = (
    "How to handle missing samples in metadata. 'error' will fail if missing "
    "samples are detected. 'ignore' will cause the feature table and metadata "
    "to be filtered, so that only samples found in both files are retained."
)
SAMPLE_CLASS_DESC = (
    "Sample classifier trained with fit_classifier."
)
SAMPLE_REGR_DESC = (
"Sample regressor trained with fit_regressor."
)


@click.group()
class cli:
    pass


@cli.command()
@click.option('--i-table', required=True)
def classify_samples(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def confusion_matrix(i_feature_table):
    pass


@cli.command()
@click.option('--i-table', required=True, help=TABLE_DESC, type=biom.Table)
@click.option('--m-metadata-file', required=True, type=click.Path(exists=True))
@click.option('--m-metadata-column', required=True, help=METADATA_COL_DESC,
              type=str)
@click.option('--p-missing-samples', default='error',
              type=click.Choice(['error', 'ignore'], case_sensitive=False),
              help=MISSING_SAMPLES_DESC)
@click.option('--output-dir', required=True)
def fit_classifier(table, metadata_file, metadata_column, missing_samples,
                   output_dir):
    pass


@cli.command()
@click.option('--i-table', required=True, help=TABLE_DESC, type=biom.Table)
@click.option('--m-metadata-file', required=True, type=click.Path(exists=True))
@click.option('--m-metadata-column', required=True, help=METADATA_COL_DESC,
              type=str)
@click.option('--p-missing-samples', default='error',
              type=click.Choice(['error', 'ignore'], case_sensitive=False),
              help=MISSING_SAMPLES_DESC)
@click.option('--output-dir', required=True)
def fit_regressor(table, metadata_file, metadata_column, missing_samples,
                  output_dir):
    pass


@cli.command()
@click.option('--i-table', required=True, help=TABLE_DESC, type=biom.Table)
@click.option('--i-sample-estimator', required=True, help=SAMPLE_CLASS_DESC)
@click.option('--output-dir', required=True)
def predict_classification(table, sample_estimator, output_dir):
    pass


@cli.command()
@click.option('--i-table', required=True, help=TABLE_DESC, type=biom.Table)
@click.option('--i-sample-estimator', required=True, help=SAMPLE_REGR_DESC)
@click.option('--output-dir', required=True)
def predict_regression(table, sample_estimator, output_dir):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def regress_samples(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def scatterplot(i_feature_table):
    pass


@cli.command()
@click.option('--i-feature-table', required=True)
def summarize(i_feature_table):
    pass


if __name__ == '__main__':
    cli()
