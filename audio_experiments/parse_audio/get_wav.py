#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import base64, os, sys, csv
import shutil
import pandas as pd

# this function converts the csv file we export from database to audio files (.3gp and .wav)
# file: input csv file path
# a_open: whether the csv file is for open response voice sample

def make_wav(file, a_open):
    gp3_folder = "./3GPs/"
    wav_folder = "./WAVs/"
    if a_open == 'open':
        gp3_folder = "./3GPs_open/"
        wav_folder = "./WAVs_open/"
    if os.path.exists(gp3_folder):
        shutil.rmtree(gp3_folder)
    os.makedirs(gp3_folder)
    if os.path.exists(wav_folder):
        shutil.rmtree(wav_folder)
    os.makedirs(wav_folder)

    max_int = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            max_int = int(max_int / 10)
            decrement = True

    with open(file) as csv_file:
        next(csv_file)

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            filename = row[0]
            if filename:
                content = row[2]
                gp3_file = gp3_folder + filename + '.3gp'
                wav_file = wav_folder + filename + '.wav'

                decoded_bytestream = base64.b64decode(content)

                fh = open(gp3_file, "wb")
                fh.write(decoded_bytestream)
                fh.close()

                os.system("ffmpeg.exe -y -i {0} {1}".format(gp3_file, wav_file))
