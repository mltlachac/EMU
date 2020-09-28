#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import csv
import get_demographic
import os 
#adds demographic numerical values to features csv files
def main(a_open):
    demographic_dict = get_demographic.get_demographic_data()
    
    # files_to_append_to = ["feature_extraction_audio/openSMILE/open_smile_features_closed.csv"]
    # files_to_append_to = ["machine_learning/experiments/e006/ids.csv"]
    files_to_append_to = ["feature_extraction_text/labeled_audio_emu.csv"]

    if a_open == 'open':
        files_to_append_to = ["feature_extraction_audio/openSMILE/open_smile_features_open.csv"]

    for file_path in files_to_append_to:
        with open(file_path, 'r') as csvinput:
            # with open(file_path[:len(file_path) - 4] + "_demographic.csv", 'w+') as csvoutput:
            with open(file_path[:len(file_path) - 4] + "_gender.csv", 'w+') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                reader = csv.reader(csvinput)

                all = []
                row = next(reader)
                # row.append('age')
                row.append('gender')
                # row.append('education')
                # row.append('student')
                all.append(row)

                for row in reader:
                    demographic = demographic_dict[int(row[0])]

                    # row.append(demographic[0]) #age
                    row.append(demographic[1]) #gender
                    # row.append(demographic[2]) #education
                    # row.append(demographic[3]) #student

                    all.append(row)

                writer.writerows(all)

# main('not')
# main('open')