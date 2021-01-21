#code authors: Ermal Toto, ML Tlachac & Joshua Lovering
#paper title: 'EMU: Early Mental Health Uncovering' 
#paper accessible at: 
#github: https://github.com/mltlachac/EMU

import sqlite3 as sql
 
# helper file for add_demographic_to_features.py
def extract_demographic(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# need to change the db file here
# use phonedata.db for old data
# use phonedata2.db for new data
# connection = sql.connect('../../database/phonedata2.db',timeout=10)
connection = sql.connect('database/phonedata2.db')
# connection = sql.connect('database/phonedata.db')


connection.row_factory = extract_demographic

cursor = connection.cursor()

cursor.execute("SELECT * FROM data WHERE type = 'demographic'")

results = cursor.fetchall()

connection.close()

def get_demographic(demographic, question):
    # separate each set of Q/A
    first_split = demographic.split(',')
    second_split = []
    list_of_answers = []
    bad_chars = ['\"', '}']
    # for each Q/A seperate the Q from the A
    for value in first_split:
        second_split.append(value.split(':'))
    # remove each Q to leave a list of answers
    # each answer is a list with one item
    for sublist in second_split:
        sublist.pop(0)
    # take each answer value and append to an empty list
    # this results in a list of answers
    for sublist in second_split:
        for value in sublist:
            for char in bad_chars:
                value = value.replace(char, '')
            list_of_answers.append(value)
    if question == "age":
        age = list_of_answers[0]
        if age == "18-22":
            target = 0
        elif age == "23-38":
            target = 1
        elif age == "39-54":
            target = 2
        elif age == "55-73":
            target = 3
        return target
    elif question == "gender":
        gender = list_of_answers[1]
        if gender == "Man":
            target = 0
        if gender == "Woman":
            target = 1
        return target
    elif question == "education":
        education = list_of_answers[2]
        if education == "Less than high school diploma":
            target = 0
        if education == "High school degree or equivalent":
            target = 1
        if education == "Some college":
            target = 2
        if education == "College degree":
            target = 3
        return target
    elif question == "student":
        student = list_of_answers[3]
        if student == "No":
            target = 0
        if student == "Yes":
            target = 1
        return target

def get_demographic_data():
    data = results
    dict = {}

    for x in data:
        demographic = []
        demographic.append(get_demographic(x['content'], "age"))
        demographic.append(get_demographic(x['content'], "gender"))
        demographic.append(get_demographic(x['content'], "education"))
        demographic.append(get_demographic(x['content'], "student"))

        if x['id'] != '':
            dict[int(x['id'])] = demographic

    return dict
