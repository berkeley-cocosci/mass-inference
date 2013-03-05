import numpy as np
import sqlite3 as sql
import os
import csv
from time import strptime

def get_completed(data_db):
    conn = sql.connect(data_db)
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT pid,condition,completion_code FROM Participants")
        vals = cur.fetchall()
    good = [(x[0], str(x[1]), str(x[2])) for x in vals if x[2] != None]
    return good

base = 'data/human'
data_dirs = [
    os.path.join(base, x) for x in os.listdir(base) 
    if x.startswith("raw_data")]
batch_dir = os.path.join(base, "batch_results")
turkid_field = "WorkerId"
code_field = "Answer.validation_code"
time_field = "AcceptTime"
time_fmt = "%a %b %d %H:%M:%S %Z %Y"
save_path = os.path.join(base, "valid_pids.npy")

batch_times = []
batch_codes = []
batch_turkids = []

print "Parsing batch files..."
batch_files = os.listdir(batch_dir)
for bf in batch_files:
    with open(os.path.join(batch_dir, bf), "r") as fh:
        dr = csv.DictReader(fh)#, delimeter=",", quotechar="\"")
        for row in dr:
            turkid = row[turkid_field]
            code = row[code_field]
            time = strptime(row[time_field], time_fmt)
            batch_times.append(time)
            batch_codes.append(code)
            batch_turkids.append(turkid)

print "Finding first use of each worker id..."
unique_ids = sorted(set(batch_turkids))
print "  --> There are %d unique worker ids" % len(unique_ids)
id_info = {}
valid_codes = []
for idx, id in enumerate(batch_turkids):
    if id not in id_info:
        id_info[id] = []
    id_info[id].append((batch_times[idx], batch_codes[idx]))
for id in unique_ids:
    info = sorted(id_info[id])
    if len(info[0][1]) == 40:
        valid_codes.append(info[0][1])
assert len(set(valid_codes)) == len(valid_codes)

print "Looking up validation codes in database..."
valid_pids = []
for data_dir in data_dirs:
    data = get_completed(os.path.join(data_dir, "data.db"))
    for pid, cond, code in data:
        if code in valid_codes:
            valid_pids.append((pid, cond))

valid_pids = np.array(sorted(set(valid_pids)))
print "  --> There are %d valid pids" % valid_pids.shape[0]

np.save(save_path, valid_pids)
print "Saved to '%s'" % save_path
