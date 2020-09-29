# 'EMU: Early Mental Health Uncovering' 

For updated information about this research, visit https://emutivo.wpi.edu/

Welcome to the repo for the EMU paper.

To install the required python packages for the programs in this repo:
1. change working directory to the repo
2. pip install requirements.txt

## Machine Learning Pipeline (emu-workflow-diagram.jpg)

* *run_feature_extraction.py*
    - this file runs *feature_extraction/feature_extraction.py*

* *feature_extraction/feature_extraction.py*
    - this file is called by *run_feature_extraction.py* and extracts features based on parameters given to main function

* *machine_learning/run_models.py*
    - this file runs machine learning experiments based on parsed arguments
    - for example run configurations, review one of these scripts
        - run_machine_learning.sh (linux)
        - run_machine_learning.bat (windows)

    - following are the possible arguments:

    --data - REQUIRED ARGUMENT
    ### path to feature csv file
        dir/example_feature_file.csv
        feature_extraction_audio/openSMILE/open_smile_features_structured.csv

    --modality - REQUIRED ARGUMENT
    ### name of modality used for output file naming
        example_feature_datatype
        audio_structured

    --feature_selection
    ### feature selection method
        pca - default
        chi2
        etc

    --cross_validation
    ### cross validation method
        loo - default
        tts

    --sampling
    ### sampling method
        regular_oversampling - default
        smote
        regular_undersampling

    --target_type
    ### target type to train and classify
        phq - default
        gad
        q9

    --oppo_target
    ### if called, the opposite target will be added as a feature after feature selection
        stores True if called

    --split
    ### integer value to split phq scores into classes
        10 - default

    --target_to_classes
   ### if called targets will not be split into binary classes
        stores False if called
