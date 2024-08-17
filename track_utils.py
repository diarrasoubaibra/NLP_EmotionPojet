# Charger le module MySQL
import mysql.connector

# Connexion à la base de données MySQL
def get_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="emotions_db"
    )
    return conn

# Fonction pour créer et gérer les tables et les enregistrements
def create_page_visited_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pageTrackTable(
            pagename VARCHAR(255),
            timeOfvisit TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_page_visited_details(pagename, timeOfvisit):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO pageTrackTable(pagename, timeOfvisit)
        VALUES(%s, %s)
    ''', (pagename, timeOfvisit))
    conn.commit()
    conn.close()

def view_all_page_visited_details():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM pageTrackTable')
    data = c.fetchall()
    conn.close()
    return data

def create_emotionclf_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS emotionclfTable(
            rawtext TEXT,
            prediction VARCHAR(255),
            probability FLOAT,
            timeOfvisit TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_prediction_details(rawtext, prediction, probability, timeOfvisit):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit)
        VALUES(%s, %s, %s, %s)
    ''', (rawtext, prediction, probability, timeOfvisit))
    conn.commit()
    conn.close()

def view_all_prediction_details():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM emotionclfTable')
    data = c.fetchall()
    conn.close()
    return data
