#!/usr/bin/python

import cgi
import cgitb
import json
import logging
import os
import httplib
import random
import shutil

from os import environ
from hashlib import sha1

import db_tools as dbt

# configure logging
logging.basicConfig(
    filename="logs/experiment.log", 
    # level=logging.WARNING,
    level=logging.DEBUG,
    format='%(levelname)s %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')

# enable debugging
cgitb.enable(display=0, logdir="logs/", format='plain')


#################
# Configuration

F_TRAINING = True
F_EXPERIMENT = True
F_POSTTEST = True
F_CHECK_IP = dbt.F_CHECK_IP

DATA_DIR = "data"
CONF_DIR = "config"

KEYWORDS = ("finished training", "finished experiment", "finished posttest")
FIELDS = ("trial", "stimulus", "response", "time", "angle", "ttype")
PFORMAT = "%03d"


#################
# Experiment functions

def validate(form):
    # try to get the participant's id
    try:
        pid = int(form.getvalue("pid"))
    except:
        logging.debug("couldn't get pid")
        return None

    # try to get the validation code
    try:
        validation_code = str(form.getvalue("validationCode"))
    except:
        logging.debug("couldn't get validation code")
        return None

    logging.debug("pid is %s and validation code is %s" % ((PFORMAT % pid), validation_code))

    # check in the database
    valid = dbt.validate_participant(pid, validation_code)
    if not valid:
        logging.debug("pid/validation code don't match")
        return None
    
    # check that the data file exists
    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))
    if not os.path.exists(datafile):
        logging.debug("datafile doesn't exist")
        return None
    
    return pid, validation_code

def parse_trialnum(t):
    if t == "trial":
        trialnum = -1
    elif t in KEYWORDS:
        trialnum = None
    else:
        trialnum = int(t)
    return trialnum
    
def get_trialnum(pid):
    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))
    with open(datafile, "r") as fh:
        data = fh.read()
    lines = data.strip().split("\n")
    i = 0
    while True:
        trial = parse_trialnum(lines[-(i+1)].split(",")[0])
        if trial is not None:
            trial += i
            break
        i += 1
    return trial
    
def create_datafile(ip_address):
    p = dbt.add_participant(ip_address)
    if p is None:
        return p

    pid, validation_code, condition = p
    logging.info("(%s) In condition %s" % ((PFORMAT % pid), condition))

    clist = os.path.join(CONF_DIR, "%s_trials.json" % condition)
    plist = os.path.join(DATA_DIR, "%s_trials.json" % (PFORMAT % pid))
    logging.info("(%s) Copying trial list to '%s'" % ((PFORMAT % pid), plist))
    shutil.copy(clist, plist)

    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))
    logging.info("(%s) Creating data file: '%s'" % ((PFORMAT % pid), datafile))
    with open(datafile, "w") as fh:
        fh.write(",".join(FIELDS) + "\n")

    return pid, validation_code
        
def write_data(pid, data):
    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))

    # write csv headers to the file if they don't exist
    if not os.path.exists(datafile):
        raise IOError("datafile does not exist: %s" % datafile)

    # write data to the file
    vals = ",".join([str(data[f]) for f in FIELDS]) + "\n"
    logging.info("(%s) Writing data to file '%s': %s" % (
        (PFORMAT % pid), datafile, vals))
    with open(datafile, "a") as fh:
        fh.write(vals)

def get_all_trialinfo(pid):
    triallist = os.path.join(DATA_DIR, "%s_trials.json" % (PFORMAT % pid))
    with open(triallist, "r") as fh:
        trialinfo = json.load(fh)
    return trialinfo
    
def get_trialinfo(pid, index):
    trialinfo = get_all_trialinfo(pid)
    if index >= len(trialinfo):
        return None
    return trialinfo[index]

def get_training_playlist(pid):
    trialinfo = get_all_trialinfo(pid)
    playlist = []
    for trial in trialinfo:
        if trial == "finished training":
            break
        playlist.append(trial['stimulus'])
    return playlist

def get_experiment_playlist(pid):
    trialinfo = get_all_trialinfo(pid)
    playlist = []
    for trial in trialinfo[trialinfo.index("finished training")+1:]:
        if trial == "finished experiment":
            break
        playlist.append(trial['stimulus'])
    return playlist

def get_posttest_playlist(pid):
    trialinfo = get_all_trialinfo(pid)
    playlist = []
    for trial in trialinfo[trialinfo.index("finished experiment")+1:]:
        if trial == "finished posttest":
            break
        playlist.append(trial['stimulus'])
    return playlist    
        
def gen_completion_code(pid):
    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))
    with open(datafile, "r") as fh:
        data = fh.read()
    code = sha1(data).hexdigest()
    dbt.set_completion_code(pid, code)
    return code

    
#################
# Http functions

def http_status(code, name):
    msg = httplib.responses[code]
    if msg != name:
        raise ValueError("invalid code and/or name: %d %s" % (code, name))
    response = "Status: %d %s\n\n" % (code, name)
    return response

def http_content_type(mime):
    response = "Content-Type: %s\n\n" % mime
    return response

    
#################
# Server functions
    
def error(msg):
    # log the error
    logging.error("Invalid request: %s" % msg)

    # respond
    print http_status(400, "Bad Request")
    print msg

def initialize(form):
    # get ip address
    ip_address = cgi.escape(environ["REMOTE_ADDR"])

    # get pid/validation code and create new data files
    info = create_datafile(ip_address)
    if info is None:
        return error("Sorry, your IP address has already been "
                     "used in this experiment.")
    else:
        pid, validation_code = info
        
    # initialization data we'll be sending
    playlist = get_training_playlist(pid)
    init = {
        'numTrials': len(playlist),
        'pid': PFORMAT % pid,
        'validationCode': validation_code,
        }
    json_init = json.dumps(init)
    
    # log information about what we're sending
    logging.info("(%s) Sending init data: %s" % 
                 ((PFORMAT % pid), json_init))

    # respond
    print http_content_type("application/json")
    print json.dumps(json_init)

def getTrialInfo(form):
    
    # make sure the pid is valid
    info = validate(form)
    if info is None:
        return error("Bad pid and/or validation code")
    pid, validation_code = info
    
    # get the index
    index = get_trialnum(pid) + 1
    
    # look up the trial information
    trialinfo = get_trialinfo(pid, index)

    if trialinfo in KEYWORDS:
        data = dict([(k, "") for k in FIELDS])
        data['trial'] = trialinfo
        write_data(pid, data)
        info = { 'index': trialinfo }

        if trialinfo == "finished training":
            playlist = get_experiment_playlist(pid)
            info['numTrials'] = len(playlist)
        elif trialinfo == "finished experiment":
            playlist = get_posttest_playlist(pid)
            info['numTrials'] = len(playlist)
        elif trialinfo == "finished posttest":
            completionCode = gen_completion_code(pid)
            info['completionCode'] = completionCode

    else:
        info = {
            'index': trialinfo['index'],
            'stimulus': trialinfo['stimulus'],
            }

    json_info = json.dumps(info)
    logging.info("(%s) Sending trial info: %s" % (
        (PFORMAT % pid), json_info))

    # respond
    print http_content_type("application/json")
    print json_info
    
def submit(form):

    # make sure the pid is valid
    info = validate(form)
    if info is None:
        return error("Bad pid and/or validation code")
    pid, validation_code = info

    # try:
    # try to extract all the relevant information
    data = dict((k, form.getvalue(k)) for k in FIELDS)
    # except:
    #     return error("Could not get all field values")

    # get the trial number
    index = get_trialnum(pid) + 1

    # populate some more data
    trialinfo = get_trialinfo(pid, index)
    data['trial'] = index
    data.update(trialinfo)

    # write the data to file
    write_data(pid, data)

    # now get the feedback
    response = {
        'feedback' : 'stable' if trialinfo['stable'] else 'unstable',
        'visual' : trialinfo['ttype'] in ("training", "posttest")
        }

    # response
    print http_content_type("application/json")
    print json.dumps(response)

    
#################
# Request handling

actions = {
    "initialize": initialize,
    "trialinfo": getTrialInfo,
    "submit": submit,
    }

# get the request
form = cgi.FieldStorage()
logging.debug("Got request: " + str(form))
for key in sorted(environ.keys()):
    logging.debug("%s %s" % (key, environ[key]))
    
# parse the action
if cgi.escape(environ['REQUEST_METHOD']) == 'POST':
    action = form.getvalue('a', None)
    logging.info("Requested action: " + action)
    handler = actions.get(action, error)
    handler(form)

else:
    error("Invalid action")
