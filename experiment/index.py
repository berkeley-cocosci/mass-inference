#!/usr/local/bin/python

import cgi
import cgitb
import json
import logging
import http_responses as http
import os
from os import environ

# configure logging
logging.basicConfig(
    filename="experiment.log", 
    level=logging.INFO,
    format='%(levelname)s %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')

# enable debugging
cgitb.enable(display=0, logdir="cgitb", format='plain')
    
#################

trials = [0, 1]
trial_types = ["catch", "normal"]

stims = ["stim_1.swf", "stim_2.swf"]
fields = ["trial", "stimulus", "question", "response", "time"]
pformat = "%03d"

questions = {
    "normal": "Will the tower fall down?",
    "catch": "Did the tower fall down?"
    }

responses = {
    "normal": [("Yes, it <b>will fall</b> down", "yes"),
               ("No, it <b>will not fall</b> down", "no")],
    "catch":  [("Yes, it <b>did fall</b>", "yes"),
               ("No, it <b>did not fall</b>", "no")]
    }

#################

def get_pid(form):
    # try to get the participant's id
    try:
        pid = int(form.getvalue("pid"))
    except:
        return None

    # check that the data file exists
    datafile = "data/%s.csv" % (pformat % pid)
    if not os.path.exists(datafile):
        return None
    
    return pid

def get_trial(pid):
    datafile = "data/%s.csv" % (pformat % pid)
    with open(datafile, "r") as fh:
        data = fh.read()
    trial = data.strip().split("\n")[-1].split(",")[0]
    trial = -1 if trial == "trial" else int(trial)
    return trial
    
def create_datafile(pid):
    datafile = "data/%s.csv" % (pformat % pid)
    logging.info("Creating data file: '%s'" % datafile)
    with open(datafile, "w") as fh:
        fh.write(",".join(fields) + "\n")

def write_data(pid, data):
    datafile = "data/%s.csv" % (pformat % pid)

    # write csv headers to the file if they don't exist
    if not os.path.exists(datafile):
        raise IOError("datafile does not exist: %s" % datafile)

    # write data to the file
    vals = ",".join([str(data[f]) for f in fields]) + "\n"
    logging.info("Writing data to file '%s': %s" % (datafile, vals))
    with open(datafile, "a") as fh:
        fh.write(vals)

#################

def error(msg):
    # log the error
    logging.error("Invalid request: %s" % msg)

    # respond
    print http.status(400, "Bad Request")
    print msg

def send_page(page):
    # read the html file
    with open(os.path.join("stages", page), "r") as fh:
        html = fh.read()

    # log information
    logging.info("Sending page '%s'" % page)
    logging.debug(html)

    # respond
    print http.content_type("text/html")
    print html

def initialize(form):
    # get list of all ids
    ids = [
        int(os.path.splitext(x)[0]) 
        for x in os.listdir("data") 
        if x.endswith(".csv")
        ]
    # set participant id (pid)
    pid = 1 if len(ids) == 0 else max(ids) + 1
    # create new data file
    create_datafile(pid)

    # initialization data we'll be sending
    init = {
        'numTrials': len(trials),
        'pid': pformat % pid
        }
    json_init = json.dumps(init)
        
    # log information about what we're sending
    logging.info("Sending init data: %s" % json_init)

    # respond
    print http.content_type("application/json")
    print json.dumps(json_init)

def getTrialInfo(form):
    
    # make sure the pid is valid
    pid = get_pid(form)
    if pid is None:
        return error("Bad pid")
    
    # get the index
    index = get_trial(pid) + 1
    logging.info("Trial %d" % index)
    
    # look up the trial information
    stim = stims[trials[index]]
    ttype = trial_types[index]
    question = questions[ttype]
    response = responses[ttype]

    info = {
        'index': index,
        'stimulus': stim,
        'question': question,
        'responses': response
        }
    json_info = json.dumps(info)

    logging.info("Sending trial info: %s" % json_info)

    # respond
    print http.content_type("application/json")
    print json_info
    
def submit(form):

    # make sure the pid is valid
    pid = get_pid(form)
    if pid is None:
        return error("Bad pid")

    try:         
        # try to extract all the relevant information
        data = dict((k, form.getvalue(k)) for k in fields)
    except:
        return error("Could not get all field values")

    # get the trial number
    index = get_trial(pid) + 1

    # populate some more data
    data['question'] = trial_types[index]
    data['trial'] = index
    data['stimulus'] = stims[trials[index]]

    # write the data to file
    write_data(pid, data)

    # respond
    print http.status(200, "OK")

#################

pages = {
    "index": "experiment.html",
    "instructions": "instructions.html",
    "trial": "trial.html",
    "finished": "finished.html",
    }

actions = {
    "initialize": initialize,
    "trialinfo": getTrialInfo,
    "submit": submit,
    }

# get the request
form = cgi.FieldStorage()
logging.info("Got request: " + str(form))
for key in sorted(environ.keys()):
    logging.debug("%s %s" % (key, environ[key]))
    
# parse the page, defaulting to the index
if environ['REQUEST_METHOD'] == 'GET':
    page = form.getvalue('page', 'index')
    logging.info("Requested page is '" + page + "'")
    send_page(pages[page])

# parse the action
elif environ['REQUEST_METHOD'] == 'POST':
    action = form.getvalue('a', None)
    logging.info("Requested action is '" + action + "'")
    handler = actions.get(action, error)
    handler(form)

else:
    error(form)
