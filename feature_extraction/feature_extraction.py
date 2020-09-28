#code authors: ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import sqlite3 as sql
import csv
import pandas as pd
import json
import os

from parse_audio import get_wav
from feature_extraction_audio.openSMILE import generate_opensmile_features as opensmile1
from feature_extraction_audio.openSMILE import arffs_to_combined_csv as opensmile2
from feature_extraction_audio.openSMILE import add_phq9_gad7_to_features as opensmile3
from feature_extraction_text import feature_extraction_text
from feature_extraction_text import add_phq9_gad7_to_features
import add_demographic_to_features

def main(type1, type2=None, database=None, ndays=14):
    # connect to database
    if database == None or database == "emu":
        conn = sql.connect('database/phonedata2.db')
        cursor = conn.cursor()
    elif database == "moodable":
        conn = sql.connect('database/phonedata.db')
        cursor = conn.cursor()

    else: assert False, f"database value unexpected: {database}"
    if type1 == 'audio':
        get_audio_features(cursor)
        get_audio_open_features(cursor)
    elif type1 == 'text':
        if type2 == 'all' or type2 == 'sent' or type2 == 'received' or type2 == 'audio':
            get_text_feature(type2, ndays, database)
        else: assert False, f"type2 value unexpected: {type2}"
    else: assert False, f"type1 value unexpected: {type1}"

#### AUDIO ####
# Audio data was taken at time of collection, so ndays is not needed
# get and clean audio data from database and save as a csv file
def get_audio_csv(cursor):
    # remove id's determined from a manual scanning of the data for illegitimate responses
    bad_audio = pd.read_csv("parse_audio/audio-summary.csv")
    bad_audio = bad_audio.loc[bad_audio['response'] == 0]
    bad_ids = list(set(bad_audio['id']))
    bad_id_str = ""
    for value in bad_ids:
        if len(str(value)) == 3:
            value = "0"+str(value)
        bad_id_str = bad_id_str+",'"+str(value)+"'"
    # remove ',' from beginning
    bad_id_str = bad_id_str[1:]

    cursor.execute("select sessionid from ids where paid = 2")
    paid_2_str = ""
    # create string containing paid id's seperarted by ,
    for row in cursor:
        id = row[0]
        paid_2_str = paid_2_str+",'"+id+"'"
    # remove ',' from beginning
    paid_2_str = paid_2_str[1:]

    # # has phq check for moodable
    # cursor.execute("select data.id from data where data.type = 'phq'")
    # has_phq_str = ""
    # # create string containing participants with phq seperarted by ,
    # for row in cursor:
    #     id = row[0]
    #     has_phq_str = has_phq_str+",'"+id+"'"
    # # remove ',' from beginning
    # has_phq_str = has_phq_str[1:]

    # cursor.execute("select distinct data.id, data.type, data.content from data where (data.type = 'audio') and (data.id in ("+has_phq_str+"))") #moodable
    # cursor.execute("select distinct data.id, data.type, data.content from data where (data.type = 'audio')") # emu everything
    # cursor.execute("select distinct data.id, data.type, data.content, ids.phoneid from data, ids where (data.type = 'audio') and (data.id = ids.sessionid) and ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu uncleaned
    cursor.execute("select distinct data.id, data.type, data.content, ids.phoneid from data, ids where (data.type = 'audio') and (data.id = ids.sessionid) and (data.id not in ("+bad_id_str+")) and ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu cleaned

    with open("feature_extraction/audio.csv", "w", newline='') as csv_file:  # Python 3 version
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description])  # write headers
        csv_writer.writerows(cursor)
    
    # # remove id's from the same phoneid
    audio_toclean = pd.read_csv("feature_extraction/audio.csv")
    audio_toclean = audio_toclean.drop_duplicates(subset="phoneid")
    audio_toclean.to_csv("feature_extraction/audio.csv", index=False)
def get_audio_open_csv(cursor):
    # remove id's determined from a manual scanning of the data for illegitimate responses
    bad_audio = pd.read_csv("parse_audio/audio_open-summary.csv")
    bad_audio = bad_audio.loc[bad_audio['response'] == 0]
    bad_ids = list(set(bad_audio['id']))
    bad_id_str = ""
    for value in bad_ids:
        if len(str(value)) == 3:
            value = "0"+str(value)
        bad_id_str = bad_id_str+",'"+str(value)+"'"
    # remove ',' from beginning
    bad_id_str = bad_id_str[1:]
    
    cursor.execute("select sessionid from ids where paid = 2")
    paid_2_str = ""
    # create string containing paid id's seperarted by ,
    for row in cursor:
        id = row[0]
        paid_2_str = paid_2_str+",'"+id+"'"
    # remove ',' from beginning
    paid_2_str = paid_2_str[1:]
    
    # cursor.execute("select distinct data.id, data.type, data.content from data where (data.type = 'audio_open')") # emu everything
    # cursor.execute("select distinct data.id, data.type, data.content, ids.phoneid from data, ids where (data.type = 'audio_open') and (data.id = ids.sessionid) and ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu uncleaned
    cursor.execute("select distinct data.id, data.type, data.content, ids.phoneid from data, ids where (data.type = 'audio_open') and (data.id = ids.sessionid) and (data.id not in ("+bad_id_str+")) and ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu cleaned
    with open("feature_extraction/audio_open.csv", "w", newline='') as csv_file:  # Python 3 version
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description])  # write headers
        csv_writer.writerows(cursor)

    # remove id's from the same phoneid
    audio_toclean = pd.read_csv("feature_extraction/audio_open.csv")
    audio_toclean = audio_toclean.drop_duplicates(subset="phoneid")
    audio_toclean.to_csv("feature_extraction/audio_open.csv", index=False)
# extract audio features and save as a csv file
def get_audio_features(cursor):
    #extract audio from database and convert to wav
    get_audio_csv(cursor)
    os.chdir("parse_audio")
    get_wav.make_wav("../feature_extraction/audio.csv", '')
    #extract openSMILE features
    os.chdir("../feature_extraction_audio/openSMILE")
    opensmile1.main('')
    opensmile2.main('')
    os.chdir("../../")
    add_demographic_to_features.main('')
    #append phq and gad to feature csv
    os.chdir("feature_extraction_audio/openSMILE")
    opensmile3.main('')
    #return to base for any future steps
    os.chdir("../../")
def get_audio_open_features(cursor):
    #extract audio from database and convert to wav
    get_audio_open_csv(cursor)
    os.chdir("parse_audio")
    get_wav.make_wav("../feature_extraction/audio_open.csv", 'open')
    #extract openSMILE features
    os.chdir("../feature_extraction_audio/openSMILE")
    opensmile1.main('open')
    opensmile2.main('open')
    os.chdir("../../")
    add_demographic_to_features.main('open')
    #append phq and gad to feature csv
    os.chdir("feature_extraction_audio/openSMILE")
    opensmile3.main('open')
    #return to base for any future steps
    os.chdir("../../")

#### TEXT ####
# get and clean text data from database and save as a csv file
# extract text features and save as a csv file
def get_text_feature(type2, ndays, database):
    os.chdir("feature_extraction_text")
    if database == "emu" or database == "moodable":
        if type2 == "all" or type2 == "sent" or type2 == "received":
            feature_extraction_text.main("text", database, ndays, type2)
            add_phq9_gad7_to_features.main("text", database, ndays, type2)
        elif type2 == 'audio':
            feature_extraction_text.main("audio", database)
            add_phq9_gad7_to_features.main("audio", database)
        else: assert False, f"type2 value unexpected: {type2}"
    else: assert False, f"database value unexpected: {database}"
    
    #return to base for any future steps or runs
    os.chdir("../../")