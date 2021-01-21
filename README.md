# 'EMU: Early Mental Health Uncovering' 

For updated information about this research, visit https://emutivo.wpi.edu/

Welcome to the repo for the EMU paper.

To install the required python packages for the programs in this repo:
1. change working directory to the /EMU
2. pip install -r requirements.txt

# Contents:

- /data_visualization
    - plot_feature_selection.py
        - output into /data_visualization/feature_selection
        - ...pca_weights.csv
            - Principal Component Analysis weights for selected features
        - ...var_PCA.csv
            - Variance distribution across selected features
        - ....json
            - name of selected features
    - TODO INSERT CORRELATION DESCRIPTION

- /EMU_data
    - /EMU_data/audio_raw
        - Raw .WAV files for structured and unstructured audio
        - manually noted summary information and transcripts
    - /EMU_data/audio_features
        - audio feature extraction .csv files using openSMILE
        - both unstructured and structured
        - both cleaned and uncleaned
            - cleaning is based on the manual review of each audio file to remove illegitimate responses.
            
- /audio_experiments
    - this directory houses feature extraction and machine learning experiments.
    - reference /audio_experiments/README.md for more information