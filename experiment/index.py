#!/usr/local/bin/python

import cgi
import cgitb
import json
import logging
import os
import httplib
import random

from os import environ
from hashlib import sha1

# configure logging
logging.basicConfig(
    filename="../experiment.log", 
    level=logging.WARNING,
    format='%(levelname)s %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')

# enable debugging
cgitb.enable(display=0, logdir="../cgitb", format='plain')


#################
# Configuration

F_TRAINING = True
F_EXPERIMENT = True
F_POSTTEST = True
F_CHECK_IP = False

html_dir = "../html"
data_dir = "../data"
conf_dir = "../config"

keywords = ["finished training", "finished experiment", "finished posttest"]
fields = ["trial", "stimulus", "question", "response", "time", "angle", "catch", "training"]
pformat = "%03d"

questions = {
    "normal": "Will the tower fall down?",
    "catch": "Did the tower fall down?"
    }

responses = {
    "normal": [("Yes, it will fall", "#FF6666", "yes"),
               ("No, it will NOT fall", "#6666FF", "no")],
    "catch": [("Yes, it did fall", "#AA0000", "yes"),
              ("No, it did NOT fall", "#0000AA", "no")]
    }


#################
# Experiment functions

def get_all_stiminfo():
    filename = os.path.join(conf_dir, "stimuli-converted.json")
    with open(filename, "r") as fh:
        stiminfo = json.load(fh)
    return stiminfo

def get_pid(form):
    # try to get the participant's id
    try:
        pid = int(form.getvalue("pid"))
    except:
        return None

    # check that the data file exists
    datafile = os.path.join(data_dir, "%s.csv" % (pformat % pid))
    if not os.path.exists(datafile):
        return None
    
    return pid

def parse_trialnum(t):
    if t == "trial":
        trialnum = -1
    elif t in keywords:
        trialnum = None
    else:
        trialnum = int(t)
    return trialnum
    
def get_trialnum(pid):
    datafile = os.path.join(data_dir, "%s.csv" % (pformat % pid))
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
    # if trial is None:
    #     trial = parse_trialnum(lines[-2].split(",")[0]) + 1
    return trial
    
def create_datafile(pid):
    datafile = os.path.join(data_dir, "%s.csv" % (pformat % pid))
    logging.info("(%s) Creating data file: '%s'" % ((pformat % pid), datafile))
    with open(datafile, "w") as fh:
        fh.write(",".join(fields) + "\n")
        
def write_data(pid, data):
    datafile = os.path.join(data_dir, "%s.csv" % (pformat % pid))

    # write csv headers to the file if they don't exist
    if not os.path.exists(datafile):
        raise IOError("datafile does not exist: %s" % datafile)

    # write data to the file
    vals = ",".join([str(data[f]) for f in fields]) + "\n"
    logging.info("(%s) Writing data to file '%s': %s" % (
        (pformat % pid), datafile, vals))
    with open(datafile, "a") as fh:
        fh.write(vals)

def create_triallist(pid):
    triallist = os.path.join(data_dir, "%s_trials.json" % (pformat % pid))
    logging.info("(%s) Creating trial list: '%s'" % ((pformat % pid), triallist))
    stiminfo = get_all_stiminfo()
    train = [stim for stim in stiminfo.keys() if stiminfo[stim]['training']]
    stims = [stim for stim in stiminfo.keys() if not stiminfo[stim]['training']]
    random.shuffle(train)
    random.shuffle(stims)
    todump = []

    # training
    if F_TRAINING:
        i = 0
        for stim in train:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, index=i)
            todump.append(info)
            i += 1
    todump.append("finished training")

    # experiment
    if F_EXPERIMENT:
        i = 0
        for stim in stims:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, index=i)
            todump.append(info)
            i += 1
    todump.append("finished experiment")

    # posttest
    if F_POSTTEST:
        i = 0
        for stim in train:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, index=i, posttest=True)
            todump.append(info)
            i += 1
    todump.append("finished posttest")

    with open(triallist, "w") as fh:
        json.dump(todump, fh)
    return len(train), len(stims)

def get_trialinfo(pid, index):
    triallist = os.path.join(data_dir, "%s_trials.json" % (pformat % pid))
    with open(triallist, "r") as fh:
        trialinfo = json.load(fh)
    if index >= len(trialinfo):
        return None
    return trialinfo[index]

def record_ip():
    ip = cgi.escape(environ["REMOTE_ADDR"])
    filename = os.path.join(data_dir, "ip_addresses.txt")
    with open(filename, "a") as fh:
        fh.write("%s\n" % ip)
    logging.info("Wrote '%s' to list of IP addresses" % ip)

def check_ip_exists():
    ip = cgi.escape(environ["REMOTE_ADDR"])
    filename = os.path.join(data_dir, "ip_addresses.txt")
    if os.path.exists(filename):
        with open(filename, "r") as fh:
            ips = [x for x in fh.read().strip().split("\n") if x != ""]
        exists = ip in ips
    else:
        exists = False
    return exists   
    
def gen_validation_code(pid):
    datafile = os.path.join(data_dir, "%s.csv" % (pformat % pid))
    with open(datafile, "r") as fh:
        data = fh.read()
    code = sha1(data).hexdigest()
    filename = os.path.join(data_dir, "%s_code.txt" % (pformat % pid))
    with open(filename, "w") as fh:
        fh.write(code)
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

def send_page(page):
    # read the html file
    with open(os.path.join(html_dir, page), "r") as fh:
        html = fh.read()

    # log information
    logging.info("Sending page '%s'" % page)
    logging.debug(html)

    # respond
    print http_content_type("text/html")
    print html

def initialize(form):
    if F_CHECK_IP:
        # check to make sure the ip address doesn't exist
        if check_ip_exists():
            error("Sorry, your IP address has already been "
                  "used in this experiment.")
            return

        # record the ip address
        record_ip()

    # get list of all ids
    ids = [
        int(os.path.splitext(x)[0]) 
        for x in os.listdir(data_dir) 
        if x.endswith(".csv")
        ]
    # set participant id (pid)
    pid = 1 if len(ids) == 0 else max(ids) + 1
    # create new data file and trial list
    create_datafile(pid)
    numtrain, numexp = create_triallist(pid)

    # initialization data we'll be sending
    init = {
        'numTraining': numtrain,
        'numExperiment': numexp,
        'pid': pformat % pid
        }
    json_init = json.dumps(init)
    
    # log information about what we're sending
    logging.info("(%s) Sending init data: %s" % 
                 ((pformat % pid), json_init))

    # respond
    print http_content_type("application/json")
    print json.dumps(json_init)

def getTrialInfo(form):
    
    # make sure the pid is valid
    pid = get_pid(form)
    if pid is None:
        return error("Bad pid")
    
    # get the index
    index = get_trialnum(pid) + 1
    
    # look up the trial information
    trialinfo = get_trialinfo(pid, index)

    if trialinfo in keywords:
        data = dict([(k, "") for k in fields])
        data['trial'] = trialinfo
        write_data(pid, data)
        info = { 'index': trialinfo }
        
        if trialinfo == "finished posttest":
            code = gen_validation_code(pid)
            info['code'] = code

    else:
        ttype = 'catch' if trialinfo['catch'] else 'normal'
        question = questions[ttype]
        response = responses[ttype]
        
        info = {
            'index': trialinfo['index'],
            'stimulus': trialinfo['stimulus'],
            'question': question,
            'responses': response,
            'training': trialinfo['training'],
            }

    json_info = json.dumps(info)
    logging.info("(%s) Sending trial info: %s" % (
        (pformat % pid), json_info))

    # respond
    print http_content_type("application/json")
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
    index = get_trialnum(pid) + 1

    # populate some more data
    trialinfo = get_trialinfo(pid, index)
    ttype = 'catch' if trialinfo['catch'] else 'normal'
    question = questions[ttype]
    data['question'] = question
    data['trial'] = index
    data.update(trialinfo)

    # write the data to file
    write_data(pid, data)

    # now get the feedback
    stable = 'undefined' if trialinfo['catch'] else trialinfo['stable']
    vfb = ('undefined' if (trialinfo['catch'] or 
                           not trialinfo['training'] or
                           trialinfo.get('posttest', False))
            else "%s-fb" % trialinfo['stimulus'])

    # response
    print http_content_type("application/json")
    print json.dumps([stable, vfb])

    
#################
# Request handling

pages = {
    "index": "experiment.html",
    }

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
    
# parse the page, defaulting to the index
if cgi.escape(environ['REQUEST_METHOD']) == 'GET':
    page = form.getvalue('page', 'index')
    logging.info("Requested page: " + page)
    send_page(pages[page])

# parse the action
elif cgi.escape(environ['REQUEST_METHOD']) == 'POST':
    action = form.getvalue('a', None)
    logging.info("Requested action: " + action)
    handler = actions.get(action, error)
    handler(form)

else:
    error(form)
