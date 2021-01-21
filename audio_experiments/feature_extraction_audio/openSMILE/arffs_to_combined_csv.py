#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import os
import ntpath
import csv

# found toCsv function here:
# https://github.com/haloboy777/arfftocsv/blob/master/arffToCsv.py

def main(a_open):
    # path = "C:/Users/lover/workspace/projects/EMU-summer-2020/EMU2020/Summer2020/feature_extraction_audio/openSMILE/"
    arff_directory = "ARFF_NEW"

    generated_features_file = "open_smile_features_closed.csv"

    # Getting all the arff files from the current directory
    files_to_convert = [arff for arff in os.listdir('ARFF_NEW') if arff.endswith(".arff")]
    if a_open == 'open':
        arff_directory = "ARFF_NEW_open"

        generated_features_file = "open_smile_features_open.csv"

        # Getting all the arff files from the current directory
        files_to_convert = [arff for arff in os.listdir('ARFF_NEW_open') if arff.endswith(".arff")]


    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    # Function for converting arff list to csv list
    def toCsv(content):
        data = False
        header = ""
        newContent = []
        for line in content:
            if not data:
                if "@attribute" in line:
                    attri = line.split()
                    columnName = attri[attri.index("@attribute") + 1]
                    header = header + columnName + ","
                elif "@data" in line:
                    data = True
                    header = header[:-1]
                    header += '\n'
                    newContent.append(header)
            else:
                newContent.append(line)
        return newContent

    # Main loop for reading and writing files
    def write_files():
        header_string = "" 
        rows = []
        for file in files_to_convert:
            with open(arff_directory + "/" + file, "r") as inFile:
                content = inFile.readlines()
                name, ext = os.path.splitext(path_leaf(inFile.name))
                new = toCsv(content)
                del new[-2]
                # print(new)
                header = new[0].split(",")
                features = new[1].split(",")

                # only get header once
                if name == "122" or name == "0122" or name == "114" or name == "0114":
                    del header[0]
                    header.insert(0, "id")
                    header_string = ','.join(header)
                # delete first column called "no name" in both header and rows
                del features[0]
                
                features.insert(0, name)
                rows.append(','.join(features))
        rows.insert(0, header_string)
        with open(generated_features_file, "w") as outFile:
            outFile.writelines(rows)

    write_files()

# main('not')
# main('open')