#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 12:00:00
module load python/gcc-8.2.0/3.7.6
source ../emu/bin/activate
python -W ignore run_models.py \
--data data/open_smile_features_open_cleaned_gender_phq_gad.csv \
--modality emu_audio_open_cleaned_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--split 1 \
--target_type q9
python -W ignore run_models.py \
--data data/open_smile_features_open_cleaned_gender_phq_gad.csv \
--modality emu_audio_open_cleaned_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--split 1 \
--target_type q9
python -W ignore run_models.py \
--data data/open_smile_features_open_cleaned_gender_phq_gad.csv \
--modality emu_audio_open_cleaned_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--split 1 \
--target_type q9
