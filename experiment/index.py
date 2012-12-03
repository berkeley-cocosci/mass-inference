#!/usr/local/bin/python

import cgi
import cgitb
import json
import logging
import http_responses as http

# configure logging
logging.basicConfig(
    filename="experiment.log", 
    level=logging.INFO,
    format='%(levelname)s %(asctime)s -- %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p')

# enable debugging
cgitb.enable()
    
#################

def index(form):
    """Respond with the physics-experiment.html main page"""

    page = "physics-experiment.html"
    with open(page, "r") as fh:
        html = fh.read()

    logging.info("Sending index page '%s'" % page)
    logging.debug(html)
    
    print http.content_type("text/html")
    print html

def start(form):
    """Shuffle the stimuli and send the list to the client"""

    stims = ["stim_1.swf", "stim_2.swf"]
    json_stims = json.dumps(stims)
    
    logging.info("Sending stimuli list: %s" % stims)
    logging.debug(json_stims)

    print http.content_type("application/json")
    print json_stims
    
def submit(form): 
    """Parse the client participant data and save it, then generate a
    Turk id and send it back"""
    
    print http.status(200, "OK")

#################

# get the request
form = cgi.FieldStorage()
logging.info("Got request: " + str(form))

# parse the action, defaulting to the index
action = form.getvalue('f', 'index')
logging.info("Requested action is '" + action + "'")

if action == "index":
    index(form)
elif action == "start":
    start(form)
elif action == "submit":
    submit(form)

else:
    logging.error("Invalid action: " + action)
    print http.status(400, "Bad Request")

