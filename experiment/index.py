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
cgitb.enable()
    
#################

def send_page(page):

    with open(os.path.join("stages", page), "r") as fh:
        html = fh.read()
    logging.info("Sending page '%s'" % page)
    logging.debug(html)

    print http.content_type("text/html")
    print html

#################
    
def initialize(form):
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
for key in sorted(environ.keys()):
    logging.info("%s %s" % (key, environ[key]))

# parse the page, defaulting to the index
if environ['REQUEST_METHOD'] == 'GET':
    page = form.getvalue('page', 'index')
    logging.info("Requested page is '" + page + "'")

    if page == "index":
        send_page("experiment.html")
    elif page == "instructions":
        send_page("instructions.html")
    elif page == "trial":
        send_page("normal-trial.html")
    elif page == "finished":
        send_page("finished.html")

# parse the action
elif environ['REQUEST_METHOD'] == 'POST':
    action = form.getvalue('a', None)
    logging.info("Requested action is '" + action + "'")

    if action == "initialize":
        initialize(form)
    elif action == "submit":
        submit(form)
    else:
        logging.error("Invalid action: " + action)
        print http.status(400, "Bad Request")

else:
    logging.error("Invalid request")
    print http.status(400, "Bad Request")

