python -W ignore run_models.py \
--data feature_extraction_audio/openSMILE/open_smile_features_closed_with_phq_gad.csv \
--modality audio_closed \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type phq
