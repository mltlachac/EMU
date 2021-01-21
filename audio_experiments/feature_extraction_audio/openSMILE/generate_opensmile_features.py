#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import os
import subprocess

def main(open):
    source = "../../parse_audio/WAVs/"
    destination = "ARFF_NEW/"
    if open == 'open':
        source = "../../parse_audio/WAVs_open/"
        destination = "ARFF_NEW_open/"
    # empty directory before running
    filelist = [f for f in os.listdir(destination) if f.endswith(".arff")]
    for f in filelist:
        os.remove(os.path.join(destination, f))
    # create directory at destination path
    os.makedirs(destination, exist_ok=True)

    for file in os.listdir(source):
        if file.endswith(".wav"):
            file_name, file_extension = os.path.splitext(file)
            wav_file = os.path.abspath(source + file)
            arff_file = destination + file_name + ".arff"
            print(wav_file)
            smil_extract_path = os.getcwd() + r"/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe"
            config_path = os.getcwd() + r"/opensmile-2.3.0/config/emobase2010.conf"
            os.system(
                "{0} -C {1} -I {2} -O {3}".format(smil_extract_path, config_path, wav_file, arff_file))

            #subprocess.Popen(["sh", os.path.expanduser("~/tester.sh")]).wait()
            # print("sys command: \n""{0} -C {1} -I {2} -O {3}".format(smil_extract_path, config_path, wav_file, arff_file))
            print("OpenSmile features extracted for " + file_name)

    print("OpenSmile feature extraction complete!")

# main('open')
# main('not')
