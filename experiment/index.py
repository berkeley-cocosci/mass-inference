#!/usr/bin/python

import cgi
import cgitb
import json
import logging
import os
import httplib
import shutil

from os import environ
from hashlib import sha1

import db_tools as dbt

# configure logging
logging.basicConfig(
    filename="logs/experiment.log",
    level=logging.WARNING,
    # level=logging.DEBUG,
    format='%(levelname)s %(asctime)s -- %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')

# enable debugging
cgitb.enable(display=0, logdir="logs/", format='plain')


#################
# Configuration

F_CHECK_IP = False

DATA_DIR = "data"
CONF_DIR = "config"
HTML_DIR = "html"

KEYWORDS = ("finished training", "finished experiment",
            "finished posttest", "query ratio")
FIELDS = ("index", "trial", "stimulus", "response", "time", "angle", "ttype")
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

    logging.debug("pid is %s and validation code is %s" % (
        (PFORMAT % pid), validation_code))

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


def parse_trialindex(t):
    if t == "index":
        trialindex = 0
    else:
        trialindex = int(t) + 1
    return trialindex


def get_trialindex(pid):
    datafile = os.path.join(DATA_DIR, "%s.csv" % (PFORMAT % pid))
    with open(datafile, "r") as fh:
        data = fh.read()
    lines = data.strip().split("\n")
    index = parse_trialindex(lines[-1].split(",")[0])
    return index


def create_datafile(ip_address, condition):
    p = dbt.add_participant(ip_address, condition, F_CHECK_IP)
    if p is None:
        return p

    pid, validation_code = p
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
        if trial == "query ratio":
            continue
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
    code = sha1(str(pid) + data).hexdigest()
    dbt.set_completion_code(pid, code)
    return code


#################
# Http functions

def http_status(code):
    msg = httplib.responses[code]
    response = "Status: %d %s\n\n" % (code, msg)
    return response


def http_content_type(mime):
    response = "Content-Type: %s\n\n" % mime
    return response


#################
# Server functions

def error(msg, code=400):
    # log the error
    response = http_status(code)
    logging.error("%s: %s" % (response.strip(), msg))

    # respond
    print response
    print msg


def send_html(html):

    # log information
    logging.info("Sending html")
    logging.debug(html)

    # respond
    print http_content_type("text/html")
    print html


def initialize(form):
    # get ip address
    ip_address = cgi.escape(environ["REMOTE_ADDR"])
    condition = str(form.getvalue("condition"))

    # get pid/validation code and create new data files
    info = create_datafile(ip_address, condition)
    if info is None:
        return error("Sorry, your IP address has already been "
                     "used in this experiment.", 403)
    else:
        pid, validation_code = info

    # initialization data we'll be sending
    index = get_trialindex(pid)
    playlist = get_training_playlist(pid)
    init = {
        'numTrials': len(playlist),
        'pid': PFORMAT % pid,
        'validationCode': validation_code,
        'index': index - 1,
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
    r_index = int(form.getvalue("index"))
    index = get_trialindex(pid)
    if r_index != index:
        return error("Requested index is %s but "
                     "actual index is %s" % (r_index, index), 405)

    # look up the trial information
    trialinfo = get_trialinfo(pid, index)
    if trialinfo in KEYWORDS:
        if trialinfo != "query ratio":
            data = dict([(k, "") for k in FIELDS])
            data['index'] = index
            data['trial'] = trialinfo
            write_data(pid, data)

        info = {
            'index': index,
            'trial': trialinfo,
            }

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
            'trial': trialinfo['trial'],
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

    # extract all the relevant information
    data = dict((k, form.getvalue(k)) for k in FIELDS)

    # get the trial number
    r_index = int(form.getvalue("index"))
    index = get_trialindex(pid)
    if r_index != index:
        return error("Submit: Requested index is %s but "
                     "actual index is %s" % (r_index, index), 405)

    # populate some more data
    trialinfo = get_trialinfo(pid, index)
    if trialinfo in KEYWORDS and trialinfo != "query ratio":
        return error("Invalid trial", 405)

    if trialinfo == "query ratio":
        # write the data to file
        data['trial'] = "query ratio"
        write_data(pid, data)

        # now get the feedback
        response = {
            'feedback': None,
            'visual': None,
            'text': None,
            'index': index,
            'trial': trialinfo,
            }

    else:
        data.update(trialinfo)
        # write the data to file
        write_data(pid, data)

        visual_fb = trialinfo['visual_fb']
        text_fb = trialinfo['text_fb']

        if (not visual_fb) and (not text_fb):
            feedback = None
        else:
            feedback = 'stable' if trialinfo['stable'] else 'unstable'

        # now get the feedback
        response = {
            'feedback': feedback,
            'visual': visual_fb,
            'text': text_fb,
            'index': index,
            'trial': trialinfo['trial'],
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
    # "submitRatio" : submitRatio,
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

elif cgi.escape(environ['REQUEST_METHOD']) == "GET":
    condition = form.getvalue('cond', None)
    if condition is None:
        error("Invalid condition", 501)

    else:
        logging.info("Condition: " + condition)

        # read the html file
        with open(os.path.join(HTML_DIR, "experiment.html"), "r") as fh:
            html = fh.read()
        send_html(html % {"condition": condition})

else:
    error("Invalid action", 501)
