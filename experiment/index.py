#!/usr/local/bin/python

import cgi
import cgitb
import json

# enable debugging
cgitb.enable()

def log_write(s):
    log = open("log.txt", "a")
    log.write(str(s) + "\n")
    log.close()

def response(s):
    log_write(s)
    print s

def index():
    response("Content-Type: text/html")
    response("")
    with open("physics-experiment.html", "r") as fh:
        html = fh.read()
    response(html)

def start():
    response("Content-Type: application/json")
    response("")
    response(json.dumps(["stim_1.swf", "stim_2.swf"]))

def error():
    response("Status: 400 Bad Request")
    response("")
    
def submit():
    response("Status: 200 OK")
    response("")

form = cgi.FieldStorage()
log_write(form)
action = form.getvalue('f', 'index')

if action == "index":
    index()
elif action == "start":
    start()
elif action == "submit":
    submit()
else:
    error()

