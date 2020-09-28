python -W ignore machine_learning/run_models.py ^
--data feature_extraction_audio/openSMILE/open_smile_features_closed_gender_phq_gad.csv ^
--modality audio_closed ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection pca ^
--oppo_target ^
--target_type phq

python -W ignore machine_learning/run_models.py ^
--data feature_extraction_audio/openSMILE/open_smile_features_closed_gender_phq_gad.csv ^
--modality audio_closed ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection chi2 ^
--target_type phq

python -W ignore machine_learning/run_models.py ^
--data feature_extraction_audio/openSMILE/open_smile_features_closed_gender_phq_gad.csv ^
--modality audio_closed ^
--cross_validation loo ^
--sampling regular_oversampling ^
--feature_selection etc ^
--target_type phq