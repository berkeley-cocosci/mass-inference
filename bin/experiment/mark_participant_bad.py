#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import DATA_PATH
from path import path
import logging
import urllib2

logger = logging.getLogger('mass.experiment')


def add_auth(url, username, password):
    """Add HTTP authencation for opening urls with urllib2.

    Based on http://www.voidspace.org.uk/python/articles/authentication.shtml

    """

    # this creates a password manager
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()

    # because we have put None at the start it will always use this
    # username/password combination for urls for which `theurl` is a
    # super-url
    passman.add_password(None, url, username, password)

    # create the AuthHandler
    authhandler = urllib2.HTTPBasicAuthHandler(passman)

    # All calls to urllib2.urlopen will now use our handler Make sure
    # not to include the protocol in with the URL, or
    # HTTPPasswordMgrWithDefaultRealm will be very confused.  You must
    # (of course) use it when fetching the page though.
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)


def mark(url, worker_id, force=False):
    request = url + "?uniqueId=%s" % worker_id
    logger.info(request)

    # try to open it
    try:
        handler = urllib2.urlopen(request)
    except IOError as err:
        if getattr(err, 'code', None) == 401:
            logger.error("Server authentication failed.")
            raise err
        else:
            raise

    response = handler.read()
    logger.info(response)


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-i", "--id",
        required=True,
        help="Worker ID to mark as bad.")
    parser.add_argument(
        "-a", "--address",
        default="http://cocosci-python.dreamhosters.com",
        help="Address of the experiment server.")
    parser.add_argument(
        "-u", "--user",
        default=None,
        help="Username to authenticate to the server.")
    parser.add_argument(
        "-p", "--password",
        default=None,
        help="Password to authenticate to the server.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force all tasks to be put on the queue.")

    args = parser.parse_args()

    # prompt for the username if it wasn't given
    if args.user is None:
        username = raw_input("Username: ")
    else:
        username = args.user

    # prompt for the password if it wasn't given
    if args.password is None:
        password = raw_input("Password: ")
    else:
        password = args.password

    # create the authentication handler to the server
    url = path(args.address).joinpath("mark_bad")
    add_auth(url, username, password)

    # fetch and save the data files
    mark(url, args.id, args.force)
