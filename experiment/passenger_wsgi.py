import sys
import os
import logging

base_path = "/home/cocosci/cocosci-python.dreamhosters.com"
INTERP = os.path.join(base_path, 'bin', 'python')
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

os.chdir(os.path.join(base_path, 'experiment'))
cwd = os.getcwd()
sys.path.append(cwd)

# logging configuration
logfilename = os.path.join(cwd, 'passenger_wsgi.log')
logformat = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
loglevel = logging.DEBUG

# configure global logging
logging.basicConfig(
    filename=logfilename, format=logformat, level=loglevel)

# configure specific logger for passenger
logger = logging.getLogger("passenger_wsgi")
handler = logging.FileHandler(logfilename)
handler.setLevel(loglevel)
handler.setFormatter(logging.Formatter(logformat))
logger.addHandler(handler)
logger.setLevel(loglevel)

# start the application
logger.debug("Starting application")
import psiturk.experiment as exp
sandbox = exp.config.getboolean("Shell Parameters", "launch_in_sandbox_mode")
logger.debug("Running in sandbox mode: %s" % sandbox)
app = exp.start_app(sandbox)

# configure application and werkzeug logging as well
app.logger.setLevel(loglevel)
app.logger.addHandler(handler)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(loglevel)
werkzeug_logger.addHandler(handler)


def application(environ, start_response):
    method = environ['REQUEST_METHOD']
    uri = environ['REQUEST_URI']
    referer = environ['HTTP_REFERER']
    user_agent = environ['HTTP_USER_AGENT']
    logger.info('"%s %s" "%s" "%s"', method, uri, referer, user_agent)
    return app(environ, start_response)


# Uncomment next two lines to enable debugging
#from werkzeug.debug import DebuggedApplication
#application = DebuggedApplication(application, evalex=True)
