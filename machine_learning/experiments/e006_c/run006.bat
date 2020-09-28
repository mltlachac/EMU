python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_gad.csv ^
--modality demographic_gad ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection pca ^
--target_type phq

python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_phq.csv ^
--modality demographic_phq ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection pca ^
--target_type gad

python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_gad.csv ^
--modality demographic_gad ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection chi2 ^
--target_type phq

python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_phq.csv ^
--modality demographic_phq ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection chi2 ^
--target_type gad

python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_gad.csv ^
--modality demographic_gad ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection etc ^
--target_type phq

python -W ignore machine_learning/run_models.py ^
--data machine_learning/experiments/e006/ids_demographic_phq_gad_phq.csv ^
--modality demographic_phq ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection etc ^
--target_type gad