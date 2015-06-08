import os
c = get_config()
c.IPEngineApp.work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'analyses'))
