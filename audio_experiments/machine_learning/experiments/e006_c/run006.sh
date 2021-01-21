#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 4:00:00
module load python/gcc-8.2.0/3.7.6
source ../emu/bin/activate
python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_closed_participants.csv \
--modality demographic_gad_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_closed_participants.csv \
--modality demographic_gad_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_closed_participants.csv \
--modality demographic_gad_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_closed_participants.csv \
--modality demographic_phq_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type gad

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_closed_participants.csv \
--modality demographic_phq_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type gad

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_closed_participants.csv \
--modality demographic_phq_uncleaned_closed_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type gad

python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_open_participants.csv \
--modality demographic_gad_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_open_participants.csv \
--modality demographic_gad_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_gad_uncleaned_open_participants.csv \
--modality demographic_gad_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type phq

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_open_participants.csv \
--modality demographic_phq_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection pca \
--target_type gad

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_open_participants.csv \
--modality demographic_phq_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection etc \
--target_type gad

python -W ignore run_models.py \
--data data/ids_demographic_phq_uncleaned_open_participants.csv \
--modality demographic_phq_uncleaned_open_participants \
--cross_validation loo \
--sampling regular_oversampling \
--feature_selection chi2 \
--target_type gad