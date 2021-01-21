#code authors: ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import pandas as pd
import numpy as np
from textblob import TextBlob
import textblob as tb
import json
import sqlite3 as sql
import csv

# database = "emu" or "moodable"
# modality = "text" or "tweet"
# ndays = 14
# text_type = "all", "sent", or "received"

def main(modality, database, ndays=14, text_type=None):
    if database == "emu":
        conn = sql.connect('../database/phonedata2.db') #EMU
        cursor = conn.cursor()
        # create string containing paid id's seperarted by ','
        cursor.execute("select sessionid from ids where paid = 2")
        paid_2_str = ""
        for row in cursor:
            id = row[0]
            paid_2_str = paid_2_str+",'"+id+"'"
        # remove ',' from beginning
        paid_2_str = paid_2_str[1:]
        
    elif database == "moodable":
        conn = sql.connect('../database/phonedata.db') #Moodable
        cursor = conn.cursor()

    if modality == "text":
        if database =="emu":
            #gather id and dates from ids table
            cursor.execute("select * from ids where ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu cleaned
            with open("../feature_extraction/"+database+"_ids.csv", "w", newline='', encoding='utf-8') as csv_file:  # Python 3 version
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                csv_writer.writerows(cursor)
            #distinct call removes text from same participant with same metadata
            #gather text content from data table
            cursor.execute("select distinct data.id, data.type, data.content from data, ids where (data.type = 'text') and (data.id = ids.sessionid) and ((ids.paid = 0 and ids.sessionid not in ("+paid_2_str+")) or ids.paid = 2)") # emu cleaned
            with open("../feature_extraction/"+database+"_textdata.csv", "w", newline='', encoding='utf-8') as csv_file:  # Python 3 version
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                csv_writer.writerows(cursor)
        elif database == "moodable":
            #gather id and dates from ids table
            cursor.execute("select * from ids")
            with open("../feature_extraction/"+database+"_ids.csv", "w", newline='', encoding='utf-8') as csv_file:  # Python 3 version
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                csv_writer.writerows(cursor)
            cursor.execute("select distinct data.id, data.type, data.content from data where data.type = 'text'")
            with open("../feature_extraction/"+database+"_textdata.csv", "w", newline='', encoding='utf-8') as csv_file:  # Python 3 version
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                csv_writer.writerows(cursor)

    if modality == 'text' or modality == 'tweet':
        #Load data
        dft2 = pd.read_csv('../feature_extraction/'+database+'_textdata.csv', encoding = "utf-8")
        dft2 = dft2.dropna().reset_index()
        print("data: " + str(dft2.shape))
        #attach date data collected
        dfids2start = pd.read_csv('../feature_extraction/'+database+'_ids.csv', encoding = "utf-8")
        dfids2 = pd.DataFrame()
        if database == "emu":
            dfids2["id"] = dfids2start.sessionid
        elif database == "moodable":
            dfids2["id"] = dfids2start.id
        dfids2["date"] = dfids2start.date
        dft2 = pd.merge(dfids2, dft2, on = "id")
        dft2 = dft2.reset_index()
        print("data last session: " + str(dft2.shape))

        #make ids represent dataset
        newids = []
        for e in range(0, dft2.shape[0]):
            # newids.append("e" + str(dft2["id"][e]))
            newids.append(str(dft2["id"][e]))
        #     newids.append(str(dft2["id"]))
        dft2["id"] = newids

        # dft1 = dft1.drop(["level_0"], axis = 1)
        # dft2 = dft2.drop(["paid"], axis = 1)

        #Combine datasets
        dft = dft2
        # dft = dft1.append(dft2)
        dft = dft.drop(["index"], axis = 1)
        print("Combined: " + str(dft.shape))
        #remove duplicated data instances
        dft = dft.drop_duplicates()
        # dft = dft.reset_index()
        print(dft.shape)

        #extract information from data instance metadata
        jsonExtract = []
        for i in range(0, len(dft.content)):
            jsonExtract.append(json.loads(str(dft.content[i])))
        names = list(jsonExtract[0])
        jsonDF = pd.DataFrame()
        for n in names:
            nlist = []
            for i in range(0, len(dft.content)):
                if n in list(jsonExtract[i]):
                    nlist.append(jsonExtract[i][n])
                else:
                    nlist.append("-100")
            jsonDF[n+"2"] = nlist
        dft = pd.concat([dft, jsonDF], axis = 1)#, sort = False) 

        #limit texts to all, sent, or received messages, type == 2
        print(dft.shape)
        if text_type == "all":
            pass
        elif text_type == "sent":
            dft = dft[dft["type2"] == "2"]
        elif text_type == "received":
            dft = dft[dft["type2"] == "1"]
        print(dft.shape)

        dft = dft[dft.date2 != "-1"]
        print(dft.shape)
        dft = dft.reset_index()



        #Limit data to ndays
        from datetime import datetime, timedelta
        from dateutil import parser
        indexes = []
        for i in range(0, dft.shape[0]):
            timeEnd = datetime.fromtimestamp(dft.date[i]/1000)
            timeStart = timeEnd - timedelta(days=ndays)
            timeCurrent = datetime.fromtimestamp(int(dft["date2"][i])/1000)
            diff = (timeStart-timeCurrent).days
            if diff>0: #will be dropped
                indexes.append(i)

        print(dft.shape)
        dft = dft.drop(indexes)
        # dft = dft.drop(["level_0"], axis = 1)
        # dft = dft.reset_index()
        print(dft.shape)

        print("Number of Messages: " + str(dft.shape[0]))
        print("Number of Participants: " + str(len(set(dft["id"]))))

        #By Participant
        dft = dft.drop(["level_0"], axis = 1)
        
        pDFt = pd.DataFrame()
        pID = []
        pContent = []
        nTweets = []
        # score = []
        for i in set(dft["id"]):
            tempdf = dft[dft["id"] == i].reset_index()
            pID.append(i)
        #     score.append(tempdf.scores[0])
            p = []
            for j in range(0, tempdf.shape[0]):
                p.append(tempdf["body2"][j])
            pContent.append(p)
            nTweets.append(len(p))
        pDFt["ID"] = pID
        pDFt["Content"] = pContent
        pDFt["Messages"] = nTweets
        # pDFt["phq"] = score

    elif modality == 'audio':
        temp_df = pd.read_csv("../parse_audio/audio_open-summary.csv")
        #remove illegitimate responses
        temp_df = temp_df[temp_df.response != 0].reset_index()
        #save id and transcript columns
        pDFt = temp_df[['id', 'transcript']]
        def str_to_list(transcript):
            return list([transcript])
        pDFt['transcript'] = pDFt['transcript'].apply(str_to_list)
        pDFt = pDFt.rename(columns={'id':'ID', 'transcript':'Content'})
    #tweet POS tags and sentiment

    polarity = []
    subjectivity = []
    tags = []
    for i in range(0, len(pDFt.ID)):
        polarity2 = []
        subjectivity2 = []
        tags2 = []
        for text in pDFt.Content[i]:
            T = TextBlob(str(text))
            polarity2.append(T.sentiment[0])
            subjectivity2.append(T.sentiment[1])
            for word, tag in T.tags:
                tags2.append(tag)
        tags.append(tags2)
        polarity.append(polarity2)
        subjectivity.append(subjectivity2)
    pDFt["POStags"] = tags
    pDFt["Polarity"] = polarity
    pDFt["Subjectivity"] = subjectivity

    #volume features for tweets
    words = []
    char = []
    for i in range(0, pDFt.shape[0]):
        w = []
        c = []
        for tweet in pDFt.Content[i]:
            w.append(len(tweet.split(" ")))
            c.append(len(tweet))
        words.append(w)
        char.append(c)
    pDFt["Words"] = words
    pDFt["Characters"] = char

    from empath import Empath
    import re

    #create list of all empath categories

    lexicon = Empath()
    emp = lexicon.analyze("Testing", normalize=True)
    wordlist = []
    for word, value in emp.items():
        wordlist.append(word)
    print(wordlist)

    #Empath features for Tweets

    for word in wordlist:
        pctt = []
        for i in range(0, pDFt.shape[0]):
            content = re.sub(r'[^\w\s]', '', str(pDFt.Content[i]).lower())
            lexicon = Empath()
            emp = lexicon.analyze(content, categories=[word], normalize = True)
            if emp != None:
                for key, value in emp.items():
                    pctt.append(value)
            else:
                pctt.append(0)
        pDFt[word] = pctt

    #create new category
    lexicon.create_category("text_abbreviations",["lol","ttyl","brb"], model="reddit")

    #new catergory for tweets
    pctt = []
    empatht = []
    for i in range(0, pDFt.shape[0]):
        content = re.sub(r'[^\w\s]', '', str(pDFt.Content[i]).lower())
        empatht.append(len(content.split(" ")))
        lexicon = Empath()
        
    # pDFt["text_abbreviations"] = pctt
    pDFt["WordsEmpath"] = empatht

    #get set of POS tags
    posTags = []
    for i in range(0, pDFt.shape[0]):
        for tag in pDFt.POStags[i]:
            posTags.append(tag)
    posSet = set(posTags)
    print(posSet)

    #POS tag counting for tweets

    poswordst = []
    for posList in pDFt.POStags:
        poswordst.append(len(posList))
    pDFt["WordsTags"] = poswordst
    for tag in posSet:
        cntAvg = []
        for posList in pDFt.POStags:
            counter = 0
            for item in posList:
                if item == tag:
                    counter += 1
            cntAvg.append(counter/len(posList))
        pDFt[tag] = cntAvg

    #sentiment features for tweets

    pcount = []
    ncount = []
    pstd = []
    nstd = []
    pavg = []
    navg = []
    scount = []
    sstd = []
    savg = []
    for i in range(0, pDFt.shape[0]):
        s = []
        p = []
        n = []
        for item in pDFt.Polarity[i]:
            if item > 0:
                p.append(item)
            if item < 0:
                n.append(item)
        for item in pDFt.Subjectivity[i]:
            if item > 0:
                s.append(item)
        pcount.append(len(p))
        ncount.append(len(n))
        scount.append(len(s))
        if len(p) > 0:
            pavg.append(sum(p)/len(p))
            pstd.append(np.std(p))
        else:
            pavg.append(0)
            pstd.append(0)
        if len(n) > 0:
            navg.append(sum(n)/len(n))
            nstd.append(np.std(n))
        else:
            navg.append(0)
            nstd.append(0)
        if len(s) > 0:
            savg.append(sum(s)/len(s))
            sstd.append(np.std(s))
        else:
            savg.append(0)
            sstd.append(0)
    pDFt["PositiveCnt"] = pcount
    pDFt["NegativeCnt"] = ncount
    pDFt["PositiveStd"] = pstd
    pDFt["NegativeStd"] = nstd
    pDFt["PostitiveAvg"] = pavg
    pDFt["NegativeAvg"] = navg
    pDFt["SubjectiveCnt"] = scount
    pDFt["SubjectiveStd"] = sstd
    pDFt["SubjectiveAvg"] = savg

    #Volume features for Tweets
    wsum = []
    wavg = []
    wstd = []
    csum = []
    cavg = []
    cstd = []
    unique = []

    for i in range(0, pDFt.shape[0]):
        wsum.append(sum(pDFt.Words[i]))
        wavg.append(sum(pDFt.Words[i])/len(pDFt.Words[i]))
        wstd.append(np.std(pDFt.Words[i]))
        csum.append(sum(pDFt.Characters[i]))
        cavg.append(sum(pDFt.Characters[i])/len(pDFt.Characters[i]))
        cstd.append(np.std(pDFt.Characters[i]))
        unique.append(len(list(set(pDFt.Content[i]))))

    pDFt["WordSum"] = wsum
    pDFt["WordAvg"] = wavg
    pDFt["WordStd"] = wstd
    pDFt["CharacterSum"] = csum
    pDFt["CharacterAvg"] = cavg
    pDFt["CharacterStd"] = cstd
    # pDFt["UniqueCnt"] = unique

    saveDFtv = pDFt.drop(columns = ["Content", "POStags", "Polarity", "Subjectivity", "Words", "Characters", "WordsEmpath", "WordsTags"])
    if modality == 'audio':
        saveDFtv.to_csv("unlabeled_" + modality + "_" + database + ".csv", encoding = "utf-8", index=False)
    else:
        saveDFtv.to_csv("unlabeled_" + modality + "_" + text_type + "_" + str(ndays) + "days_" + database + ".csv", encoding = "utf-8", index=False)