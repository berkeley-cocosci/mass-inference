import os, random, time
import numpy as np
import sqlite3 as sql
from hashlib import sha1

DATA_DB = "data/data.db"
BACKUP_DB = "data/data.db.bak"
CONF_DIR = "config"
F_CHECK_IP = False
    
def create():
    # back up any existing database
    if os.path.exists(DATA_DB):
        # remove old backup if it exists
        if os.path.exists(BACKUP_DB):
            print "Removing old backup..."
            os.remove(BACKUP_DB)
        print "Backing up old database to '%s'..." % BACKUP_DB
        os.rename(DATA_DB, BACKUP_DB)

    # create the new database
    print "Creating new database '%s'..." % DATA_DB
    conn = sql.connect(DATA_DB)
    with conn:
        cur = conn.cursor()
        
        # create participants table
        cur.execute("CREATE TABLE Participants(pid INTEGER PRIMARY KEY AUTOINCREMENT, validation_code TEXT, condition TEXT, ip_address TEXT, completion_code TEXT)")
        print "Created 'Participants' table"

        # create and conditions table
        cur.execute("CREATE TABLE Conditions(id TEXT)")
        print "Created 'Conditions' table"

        # populate conditions table
        lists = [x for x in os.listdir(CONF_DIR) if x.endswith("_trials.json")]
        conditions = sorted([x.split("_")[0] for x in lists])
        for condition in conditions:
            cur.execute("INSERT INTO Conditions VALUES (?)", (condition,))
            print "Inserted condition '%s'" % condition

            
def check_ip(cur, ip_address):
    cur.execute(
        "SELECT pid,validation_code FROM Participants WHERE ip_address=?",
        (ip_address,))
    pids = cur.fetchall()
    return pids

def choose_condition(cur):
    # get available conditions
    cur.execute("SELECT id FROM Conditions")
    conditions = np.array(cur.fetchall())

    # get conditions of participants that have finished
    cur.execute("SELECT condition FROM Participants WHERE NOT completion_code=NULL")
    vals = np.array(cur.fetchall())

    # count up the number of participants in each condition
    if vals.size == 0:
        counts = np.zeros(conditions.shape[0], dtype='i4')
    else:
        counts = np.sum(vals == conditions.T, axis=0)

    # randomly choose the condition out of the conditions with the
    # least number of completed participants
    mins = np.nonzero(counts == np.min(counts))[0]
    idx = random.choice(mins)
    condition = conditions[idx].ravel()[0]
    
    return condition

def add_participant(ip_address):
    conn = sql.connect(DATA_DB)
    with conn:
        cur = conn.cursor()
        
        # make sure the IP address doesn't exist already
        pids = check_ip(cur, ip_address)
        if F_CHECK_IP and len(pids) > 0:
            participant = None

        else:
            # get the condition
            condition = choose_condition(cur)
    
            # generate a unique validation code
            validation_code = sha1(ip_address + str(time.time())).hexdigest()
    
            # add a new row for this participant
            cur.execute(
                "INSERT INTO Participants VALUES (NULL, ?, ?, ?, NULL)",
                (validation_code, condition, ip_address))

            # get the new participant's info
            cur.execute(
                "SELECT * FROM Participants WHERE validation_code=?", 
                (validation_code,))
            rows = cur.fetchall()
    
            # double check that the validation code is unique...
            assert len(rows) == 1
            
            # return the pid, validation code, and condition
            participant = rows[0][:3]
            
    return participant

def validate_participant(pid, validation_code):
    conn = sql.connect(DATA_DB)
    with conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT pid FROM Participants WHERE pid=? AND validation_code=?",
            (pid, validation_code))
        vals = cur.fetchall()
        assert len(vals) <= 1
        valid = len(vals) == 1

    return valid

def set_completion_code(pid, completion_code):
    conn = sql.connect(DATA_DB)
    with conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE Participants SET completion_code=? WHERE pid=?",
            (completion_code, pid))

def list_participants():
    conn = sql.connect(DATA_DB)
    with conn:
        cur = conn.cursor()
        
        cur.execute("SELECT * FROM Participants")
        col_names = [cn[0] for cn in cur.description]
        rows = cur.fetchall()
        colstr = "%s  %-40s  %s  %-15s  %-40s" % tuple(col_names)
        print colstr
        print "-" * len(colstr)
        for row in rows:
            print "%3s  %40s  %-9s  %-15s  %-40s" % row

# # some testing code
# create()
# print

# (pid, validation_code, condition) = add_participant("192.168.0.1")
# (pid, validation_code, condition) = add_participant("192.168.0.2")
# (pid, validation_code, condition) = add_participant("192.168.0.3")
# list_participants()

# assert validate_participant(pid, validation_code)
# assert not validate_participant(pid+1, validation_code)
# assert not validate_participant(pid, validation_code + "a")


