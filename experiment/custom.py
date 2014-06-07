# this file imports custom routes into the experiment server

from flask import Blueprint, render_template, Response, abort
from jinja2 import TemplateNotFound

from psiturk.psiturk_config import PsiturkConfig
from psiturk.user_utils import PsiTurkAuthorization, nocache

# Database setup
from psiturk.models import Participant
from sqlalchemy import or_

# load the configuration options
config = PsiturkConfig()
config.load_config()

# if you want to add a password protect route use this
myauth = PsiTurkAuthorization(config)

# explore the Blueprint
custom_code = Blueprint(
    'custom_code', __name__,
    template_folder='templates',
    static_folder='static')

# Status codes
NOT_ACCEPTED = 0
ALLOCATED = 1
STARTED = 2
COMPLETED = 3
SUBMITTED = 4
CREDITED = 5
QUITEARLY = 6
BONUSED = 7


def get_participants():
    codeversion = config.get('Task Parameters', 'experiment_code_version')
    participants = Participant\
        .query\
        .filter(Participant.codeversion == codeversion)\
        .filter(or_(
            Participant.status == COMPLETED,
            Participant.status == CREDITED,
            Participant.status == SUBMITTED,
            Participant.status == BONUSED))\
        .all()
    return participants


@custom_code.route('/data/<name>', methods=['GET'])
@myauth.requires_auth
@nocache
def download_datafiles(name):
    contents = {
        "trialdata": lambda p: p.get_trial_data(),
        "eventdata": lambda p: p.get_event_data(),
        "questiondata": lambda p: p.get_question_data()
    }

    if name not in contents:
        abort(404)

    query = get_participants()
    ret = "".join([contents[name](p) for p in query])
    response = Response(
        ret,
        content_type="text/csv",
        headers={
            'Content-Disposition': 'attachment;filename=%s.csv' % name
        })

    return response
