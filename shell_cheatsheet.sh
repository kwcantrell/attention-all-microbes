# Train T1000
python gotu_cli.py sequence2sequence \
    --i-table-path /home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/asv_ordered_table.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/rotation_results/tulsa1000/gotu_ordered_table_filtered.biom \
    --i-max-bp 150 \
    --p-epochs 15 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/t1000 \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/t1000

# Train AGP
python gotu_cli.py sequence2sequence \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/agp_asv_table.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/agp_gotu_table.biom \
    --i-max-bp 150 \
    --p-epochs 15 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/agp-new \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/agp-new

#Train AGP 80/20
python gotu_cli.py sequence2sequence \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/agp_asv_train.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/agp_gotu_train.biom \
    --i-max-bp 150 \
    --p-epochs 15 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/agp_80_20 \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/agp_80_20

#Train Finrisk Full
python gotu_cli.py sequence2sequence \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/12142_asv_table.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/12142_gotu_table.biom \
    --i-max-bp 150 \
    --p-epochs 15 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/finrisk \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/finrisk

#Train Finrisk 80/20
python gotu_cli.py sequence2sequence \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/finrisk_asv_train.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/finrisk_gotu_train.biom \
    --i-max-bp 150 \
    --p-epochs 15 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/finrisk_80_20 \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/finrisk_80_20

# AGP 80/20 Test
python gotu_cli.py predict-s2s \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/agp_asv_test.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/agp_gotu_test.biom \
    --i-max-bp 150 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-fn agp_80_20 \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/predictions \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/agp_80_20

#Finrisk 80/20 Test
python gotu_cli.py predict-s2s \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/finrisk_asv_test.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/train_test_splits/finrisk_gotu_test.biom \
    --i-max-bp 150 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-fn finrisk_80_20 \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/predictions \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/finrisk_80_20

#Finrisk against AGP
python gotu_cli.py predict-s2s \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/12142_asv_table.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/12142_gotu_table.biom \
    --i-max-bp 150 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-fn finrisk_against_agp \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/predictions \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/agp-new

#AGP against Finrisk
python gotu_cli.py predict-s2s \
    --i-table-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/10317_asv_table.biom \
    --i-gotu-path /home/jokirkland/data/asv2gotu/paired_asv_gotu_data/subsets/10317_gotu_table.biom \
    --i-max-bp 150 \
    --p-base-model-path /projects/deep-learning/foundation-model-new/model.keras \
    --p-output-fn agp_against_finrisk \
    --p-output-dir /home/jokirkland/data/asv2gotu/aam_testing/predictions \
    --p-model-weights-path /home/jokirkland/data/asv2gotu/aam_testing/finrisk