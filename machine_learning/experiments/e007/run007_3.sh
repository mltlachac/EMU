#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 4:00:00
module load python/gcc-8.2.0/3.7.6
source ../emu/bin/activate
python -W ignore run_models.py \
--data data/labeled_audio_emu_gender.csv \
--modality audio_transcript_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--split 1 \
--target_type q9
python -W ignore run_models.py \
--data data/labeled_audio_emu_gender.csv \
--modality audio_transcript_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--split 1 \
--target_type q9
python -W ignore run_models.py \
--data data/labeled_audio_emu_gender.csv \
--modality audio_transcript_gender \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--split 1 \
--target_type q9