import sqlite3 as sql

def extract_phq_score(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# use phonedata.db for old data
# use phonedata2.db for new data
# connection = sql.connect('../../database/phonedata2.db',timeout=10)
connection = sql.connect('database/phonedata2.db',timeout=10)
# connection = sql.connect('database/phonedata.db',timeout=10)


connection.row_factory = extract_phq_score

cursor = connection.cursor()

cursor.execute("SELECT * FROM data WHERE type = 'phq'")

results = cursor.fetchall()

connection.close()

def get_phq_score(phq):
    return int(phq[7:8])+int(phq[16:17])+int(phq[25:26])+int(phq[34:35])+int(phq[43:44])+int(phq[52:53])+int(phq[61:62])+int(phq[70:71])+int(phq[79:80])


def get_9_score(phq):
    return int(phq[79:80])


def get_phq_data():
    data = results
    dict = {}


    for x in data:
        phq = []
        phq.append(get_9_score(x['content']))
        phq.append(get_phq_score(x['content']))


        if x['id'] != '':
            dict[int(x['id'])] = phq

    return dict
