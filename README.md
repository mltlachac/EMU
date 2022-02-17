# 'EMU: Early Mental Health Uncovering' 

The paper is available at https://ieeexplore.ieee.org/document/9680143

Regarding the paper: “© 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.”

If you use the code, data, visualizations, or paper from this repository, cite:

ML Tlachac, Ermal Toto, Joshua Lovering, Rimsha Kayastha, Nina Taurich, Elke Rundensteiner, "EMU: Early Mental Health Uncovering Framework and Dataset", 20th IEEE International Conference of Machine Learning Applications (ICMLA) Special Session Machine 
Learning in Health, 2021

```
@inproceedings{Tlachac2021EMU,
  title={EMU: Early Mental Health Uncovering Framework and Dataset},
  author={M. L. Tlachac and Ermal Toto and Joshua Lovering and Rimsha Kayastha and Nina Taurich and Elke Rundensteiner},
  booktitle={20th IEEE International Conference of Machine Learning Applications (ICMLA) Special Session Machine 
Learning in Health},
  year={2021}}
```

For updated information about this research, visit https://emutivo.wpi.edu/

Welcome to the repository for the EMU paper.

To install the required python packages for the programs in this repo:
1. change working directory to /EMU
2. pip install -r requirements.txt

## Contents:

- /EMU_data
    - /EMU_data/audio_raw
        - Raw .WAV files for structured and unstructured audio
        - manually noted summary information and transcripts
    - /EMU_data/audio_features
        - audio feature extraction .csv files using openSMILE
        - both unstructured and structured
        - both cleaned and uncleaned
            - cleaning is based on the manual review of each audio file to remove illegitimate responses.

- /data_visualization
    - plot_feature_selection.py
        - generates feature selection files
        - output into /data_visualization/feature_selection
    - /data_visualization/feature_selection
        - ...pca_weights.csv
            - Principal Component Analysis weights for selected features
        - ...var_PCA.csv
            - Variance distribution across selected features
        - ....json
            - name of selected features
    - TODO INSERT CORRELATION DESCRIPTION
  
- /audio_experiments
    - this directory houses feature extraction and machine learning experiments.
    - reference /audio_experiments/README.md for more information
