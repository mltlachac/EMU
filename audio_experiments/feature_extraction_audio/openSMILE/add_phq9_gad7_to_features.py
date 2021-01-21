#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import csv
from feature_extraction_audio.openSMILE import get_phq9 as phq
from feature_extraction_audio.openSMILE import get_gad7 as gad
import os 
  
def main(a_open):
    phq_dict = phq.get_phq_data()
    gad_dict = gad.get_gad_data()

    files_to_append_to = ["open_smile_features_closed_demographic.csv"]
    # files_to_append_to = ["open_smile_features_closed_gender.csv"]
    # files_to_append_to = ["open_smile_features_closed.csv"]

    if a_open == 'open':
        files_to_append_to = ["open_smile_features_open_demographic.csv"]
        # files_to_append_to = ["open_smile_features_open_gender.csv"]
        # files_to_append_to = ["open_smile_features_open.csv"]


    for file_path in files_to_append_to:
        # print(os.getcwd())
        with open(file_path, 'r') as csvinput:
            with open(file_path[:len(file_path) - 4] + "_phq_gad.csv", 'w+') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                reader = csv.reader(csvinput)

                all = []
                row = next(reader)
                row.append('q9')
                row.append('phq')
                row.append('gad')
                all.append(row)

                for row in reader:
                    phq_for_id = phq_dict[int(row[0])]
                    row.append(phq_for_id[0])
                    row.append(phq_for_id[1])
                    
                    gad_for_id = gad_dict[int(row[0])]
                    row.append(gad_for_id[0])

                    all.append(row)

                writer.writerows(all)

# main('not')
# main('open')