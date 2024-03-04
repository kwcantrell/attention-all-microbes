TABLE_DESC = ("Feature table containing all features that should be used for "
              "target prediction.")
METADATA_COL_DESC = "Categorical metadata column to use as prediction target. "
MISSING_SAMPLES_DESC = ("How to handle missing samples in metadata. 'error' "
                        "will fail if missing samples are detected. 'ignore' "
                        "will cause the feature table and metadata to be "
                        "filtered, so that only samples found in both files "
                        "are retained.")
SAMPLE_CLASS_DESC = "Sample classifier trained with fit_classifier."
SAMPLE_REGR_DESC = "Sample regressor trained with fit_regressor."
