#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 12:00:00
module load python/gcc-8.2.0/3.7.6
source ../emu/bin/activate
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type phq
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type phq
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type phq
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type gad
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type gad
python -W ignore run_models.py \
--data data/open_smile_features_open_uncleaned_demographic_phq_gad.csv \
--modality emu_audio_open_uncleaned_demographic \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type gad