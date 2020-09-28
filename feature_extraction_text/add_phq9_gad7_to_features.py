#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import pandas as pd
import get_phq9 as phq
import get_gad7 as gad
import os
import csv

# this function attach phq scores to coressponding participants for text features.
def main(modality, database, ndays=None, text_type=None):
    if modality == "text":
        file_path = "unlabeled_" + modality + "_" + text_type + "_" + str(ndays) + "days_" + database + '.csv'
    elif modality == 'audio':
        file_path = "unlabeled_" + modality + "_" + database + '.csv'
    # data = pd.read_csv(filename, index_col='id')
    phq_dict = phq.get_phq_data()
    gad_dict = gad.get_gad_data()
    # phqdata = pd.DataFrame.from_dict(phq, orient='index',columns=['PHQ9','PHQTotal'])

    with open(file_path, 'r') as csvinput:
        with open(file_path[2:], 'w+') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('q9')
            row.append('phq')
            row.append('gad')
            all.append(row)

            for row in reader:
                if row[0] == 'id':
                    continue
                else:
                    phq_for_id = phq_dict[int(row[0])]
                    row.append(phq_for_id[0])
                    row.append(phq_for_id[1])
                    
                    try:
                        gad_for_id = gad_dict[int(row[0])]
                        row.append(gad_for_id[0])
                    except:
                        row.append('n/a')
                all.append(row)

            writer.writerows(all)

# fandP = data.join(phqdata)

# fandP.to_csv('TextFeaturesComplete.csv')
# os.remove("headertextdatagrouped.csv")
# os.remove("textdata.csv")
# os.remove("textdatafeature.csv")
# os.remove("textdatagrouped.csv")
# os.remove("textfeaturesonly.csv")