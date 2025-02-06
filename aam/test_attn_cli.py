from attention_cli_wrapper import fit_asv_encoder, fit_denoised_unifrac_regressor, fit_taxonomy_regressor, \
    fit_sample_regressor, predict_sample_regressor, fit_gotu

from click.testing import CliRunner

def test_fit_asv_encoder():
    runner = CliRunner()
