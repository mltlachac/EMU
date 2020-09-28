import sqlite3 as sql


def extract_gad_score(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# need to change the db file here
# use phonedata.db for old data
# use phonedata2.db for new data
connection = sql.connect('database/phonedata2.db',timeout=10)
# connection = sql.connect('database/phonedata.db',timeout=10)

connection.row_factory = extract_gad_score

cursor = connection.cursor()

cursor.execute("SELECT * FROM data WHERE type = 'gad'")

results = cursor.fetchall()

connection.close()

def get_gad_score(gad):
    return int(gad[7:8])+int(gad[16:17])+int(gad[25:26])+int(gad[34:35])+int(gad[43:44])+int(gad[52:53])+int(gad[61:62])


# def get_9_score(phq):
#     return int(phq[79:80])


def get_gad_data():
    data = results
    dict = {}


    for x in data:
        gad = []
        # phq.append(get_9_score(x['content']))
        gad.append(get_gad_score(x['content']))


        if x['id'] != '':
            dict[int(x['id'])] = gad

    return dict
