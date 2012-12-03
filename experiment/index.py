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
stims = ["stim_1.swf", "stim_2.swf"]
fields = ["trial", "stimulus", "response", "time"]
turkid = "XYZ"

#################

def write_data(data):
    datafile = "data/%s.csv" % turkid

    # write csv headers to the file if they don't exist
    if not os.path.exists(datafile):
        logging.info("Creating data file: '%s'" % datafile)
        with open(datafile, "w") as fh:
            fh.write(",".join(fields) + "\n")

    # write data to the file
    vals = ",".join([str(data[f]) for f in fields]) + "\n"
    logging.info("Writing data to file '%s': %s" % (datafile, vals))
    with open(datafile, "a") as fh:
        fh.write(vals)

#################

def error(form=None):
    # log the error
    logging.error("Invalid request")

    # respond
    print http.status(400, "Bad Request")

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
    # log information about what we're sending
    logging.info("Sending number of trials: %d" % len(trials))

    # respond
    print http.content_type("application/json")
    print json.dumps(len(trials))

def getStimulus(form):
    # get the index from the form
    sindex = form.getvalue('index', 'undefined')

    # try to convert it into an integer
    try:
        index = int(sindex)
    except:
        logging.error("Bad index: %s" % sindex)
        return error()

    # look up the stimulus
    stim = stims[trials[index]]
    logging.info("Sending stimulus name: %s" % stims)

    # respond
    print http.content_type("application/json")
    print json.dumps(stim)
    
def submit(form): 
    # try to extract all the relevant information that was submitted
    try:
        data = dict((k, form.getvalue(k)) for k in fields)
    except:
        return error()

    # write the data to file
    write_data(data)

    # respond
    print http.status(200, "OK")

#################

pages = {
    "index": "experiment.html",
    "instructions": "instructions.html",
    "trial": "normal-trial.html",
    "finished": "finished.html",
    }

actions = {
    "initialize": initialize,
    "stimulus": getStimulus,
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
