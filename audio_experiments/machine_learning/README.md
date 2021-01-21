# Machine Learning Experiements

Note: For some experiements, the opposite target was added as a feature after feature selection. For example, if PHQ-9 was being used as the target, the GAD-7 score would have been added as a feature.

- e001:
    Data: EMU vs Moodable | Cleaned Structured
    Features: Audio Features + Gender
    Target: PHQ-9 Score

- e002
    Data: EMU vs Moodable | Uncleaned Structured
    Features: Audio Features
    Target: PHQ-9 Score

- e003
    Data: EMU | Uncleaned, Cleaned, Structured, Untructured
    Features: Audio Features + Gender
    Target: PHQ-9 Score, Gad-7 Score

- e003_b
    Data: EMU | Uncleaned, Cleaned, Structured, Untructured
    Features: Audio Features, Opposite Target, and Gender
    Target: PHQ-9 Score, Gad-7 Score

- e003_b
    Data: EMU | Uncleaned, Cleaned, Structured, Untructured
    Features: Audio Features + Gender
    Target: PHQ Question-9

- e004
    Same as e005
    
- e005
    Data: EMU | Uncleaned, Cleaned, Structured, Untructured
    Features: Audio Features, Age, Gender, Education, and Student Status
    Target: PHQ-9 Score, Gad-7 Score

- e005_b
    Data: EMU | Uncleaned, Cleaned, Structured, Untructured
    Features: Audio Features, Age, Gender, Education, and Student Status
    Target: PHQ-9 Score, Gad-7 Score

- e006
    Data: EMU
    Features: Opposite Target, Age, Gender, Education, Student Status
    Target: PHQ-9 Score, Gad-7 Score

- e006_b
    Data: EMU
    Features: Age, Gender, Education, Student Status
    Target: PHQ-9 Score, Gad-7 Score

- e006_c
    Data: EMU
    Features: Opposite Target, Age, Gender, Education, Student Status
    Target: PHQ-9 Score, Gad-7 Score
    Note: Only includes data from individuals who also submitted audio

- e007
    Data: EMU | Cleaned, Unstructured
    Features: Text Features from Transcript + Gender
    Target: PHQ-9 Score, PHQ Question-9, Gad-7 Score